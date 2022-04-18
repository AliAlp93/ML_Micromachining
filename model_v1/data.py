"""
Data structures and collections for various stages of processing.

# Contains:
- `MachiningDataset`: Wrapper for `torch` `Dataset` which combines all the data from a list of `ExperimentData` into a runnable `Dataset`.
- `ExampleData`: A set of example data from one machining experiment which combines metadata about a machining experiment from `ExperimentData` with a stack of all learnable data from `TrainingStack`.
- `ExperimentData`: Collection of `ChannelData` and `TrainingStack`s for one experiment (e.g. `40k-15mm-100um`).
- `TrainingStack`: Stacks of Data from across all channels to be fed into the model.
- `ChannelData`: Container for file metadata and `ChannelSection` data derived from a file.
- `ChannelSection`: Container for all processed and unprocessed data for each channel section.

**Note:** Most classes also include processing functions in their `from_mat` classmethods.
**Note:** All data is intentionally frozen (immutable) since we shouldn't be
altering it after collection except for post-processing which occurs in the `from_mat` methods.
"""
from __future__ import annotations  # Activate postponed annotations (for using classes as return type in their own methods)

from typing import List, Dict, Tuple, Union, cast
import attr

import re
import os

import scipy.io  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resample_vector
from logger import logger


class MachiningDataset(Dataset):
    """
    Wrapper for `torch` `Dataset` which combines all the data from a list of `ExperimentData` into a runnable `Dataset`.
    """
    examples: List[DataExample]

    def __init__(self, experiments: List[ExperimentData],  rough2):
        self.examples = []
        # Grab all training stacks from all experiments, augment them with
        # experiment metadata, and collect them:
        for exp in experiments:
            #for stack in range(2,3): #exp.training_stacks:
                self.examples.append(DataExample(
                    spindle_speed=exp.spindle_speed,
                    feed_rate=exp.feed_rate,
                    depth=exp.depth,
                    stack=exp.training_stacks[1],
                    Roughness=rough2[exp.dir_name].value[:,2] # 0:PV   ,1:rms    ,2:Sa the rest is standard deviations.
                ))

    def __len__(self) -> int: return len(self.examples)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Union[Dict, List[Dict]]:
        if torch.is_tensor(idx):
            indices: List[int] = cast(torch.Tensor, idx).tolist()
            if len(indices) == 1:
                return attr.asdict(self.examples[indices[0]])
            else:
                return [attr.asdict(self.examples[i]) for i in indices]
        elif isinstance(idx, int):
            return attr.asdict(self.examples[idx])
        else:
            raise TypeError(
                "Invalid `__getitem__` index type. "
                f"Expected `Tensor` or `int`, got: {type(idx)}."
            )


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class DataExample:
    """
    A set of example data from one machining experiment which combines metadata 
    about a machining experiment from `ExperimentData` with a stack of all 
    learnable data from `TrainingStack`.
    """
    spindle_speed: int  # [kHz], e.g.: 40k, 60k, 80k
    feed_rate: int  # [mm/s]
    depth: int  # [um]
    stack: TrainingStack
    Roughness: torch.Tensor

@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class ExperimentData:
    """
    Collection of `ChannelData` and `TrainingStack`s for one experiment (e.g. `40k-15mm-100um`).
    """
    dir_name: str
    spindle_speed: int  # [kHz], e.g.: 40k, 60k, 80k
    feed_rate: int  # [mm/s]
    depth: int  # [um]
    data: List[ChannelData]  # List of `ChannelData`, *sorted* by `channel_num`

    # List of TrainingStacks for feeding into the model:
    training_stacks: List[TrainingStack]

    @classmethod
    def from_mat(cls, dir_name: str, data: List[ChannelData]) -> ExperimentData:
        """
        Creates a `ChannelSection` from the data extracted from a `mat` file, given
        the name of a directory which contained `ChannelData` and an **unsorted**
        list of `ChannelData` extracted from that directory.
        """
        logger.verbose(  # type: ignore
            f"\t Collecting experiment {dir_name} . . ."
        )
        # Extract experiment metadata (parameters) from `dir_name`:
        spindle_speed, feed_rate, depth = [
            int(x) for x in re.findall(r'\d+', dir_name)
        ]
        # Sort `ChannelData` in place:
        data.sort(key=lambda cd: cd.channel_num) #WHAT IS THIS, WHAT KIND OF SORTING!!!! sorting the indices like 01 02 03 , 1 2 3 cause  1 2 3 is causing problems???

        # Create Training Stacks:
        # Get the number of sections and make sure all channels have the same number of sections:
        section_counts = [len(d.sections) for d in data]
        if any([n != section_counts[0] for n in section_counts]):
            raise ValueError(
                f"All channels in {dir_name} should have the same number of sections. "
                f"Instead, they have the following section counts (sorted by channel number): {section_counts}."
            )
        num_sections = section_counts[0]

        training_stacks: List[TrainingStack] = []

        # Dictionary to map each `TrainingStack` field to a `ChannelSection`
        # field (for all of those which are directly mapped,
        # i.e. not `signalForces`)
        stack_fields: Dict[str, str] = {
            'signalAE': 'AE_FFT_val',
            'signalMic': 'Mic_FFT_val',
            'signalFx': 'Fxcomp_mu',
            'signalFy': 'Fycomp_mu',
            'signalFz': 'Fzcomp_mu',
            'signalAE_BG': 'AE_FFT_BG_val',
            'signalMic_BG': 'Mic_FFT_BG_val',
            'signalFx_std':'Fxcomp_sig',
            'signalFy_std':'Fycomp_sig',
            'signalFz_std':'Fzcomp_sig',
            'signalSpec_Mic':'Mic_SPEC',
            'signalSpec_AE':'AE_SPEC',
            'signalSpec_Mic_BG':'Mic_SPEC_BG',
            'signalSpec_AE_BG':'AE_SPEC_BG'
            

        }

        for s in range(num_sections):
            # Dictionary to map each `TrainingStack` field to a sorted list of
            # all the Tensors for each `ChannelSection` field from all channels:
            stack_tensors: Dict[str, List[torch.Tensor]] = dict(
                (k, []) for k in stack_fields.keys()
            )

            for channel in data:
                section = channel.sections[s]
                for stack_field, section_field in stack_fields.items():
                    stack_tensors[stack_field].append(
                        getattr(section, section_field)
                    )

            # Cat all tensors in each stack:
            stacks: Dict[str, torch.Tensor] = dict(
                (k, torch.stack(ts, 0)) for k, ts in stack_tensors.items()
            )
            # Add special tensors:
            stacks['signalForces'] = torch.stack(
                (stacks['signalFx'], stacks['signalFy'], stacks['signalFz'])
            )

            training_stacks.append(TrainingStack(
                section_num=s,
                **stacks
            ))

        # Build and return `ExperimentData`:
        return cls(
            dir_name=dir_name,
            spindle_speed=spindle_speed,
            feed_rate=feed_rate,
            depth=depth,
            data=data,
            training_stacks=training_stacks
        )


@attr.s(auto_attribs=True) #frozen=True, cmp=True, slots=True, auto_attribs=True
class TrainingStack:
    """
    Stacks of Data from across all channels to be fed into the model.
    Each stack contains samples from only one section for each channel.
    """
    section_num: int
    signalAE: torch.Tensor
    signalMic: torch.Tensor
    signalFx: torch.Tensor
    signalFy: torch.Tensor
    signalFz: torch.Tensor
    signalAE_BG: torch.Tensor
    signalMic_BG: torch.Tensor
    signalFy_std: torch.Tensor
    signalFx_std: torch.Tensor
    signalFz_std: torch.Tensor
    signalSpec_Mic: torch.Tensor
    signalSpec_AE: torch.Tensor
    signalSpec_Mic_BG: torch.Tensor
    signalSpec_AE_BG: torch.Tensor
    signalForces: torch.Tensor

    
    
@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class RoughnessData:
    """Container for file metadata and `ChannelSection` data derived from a file."""
    # File metadata:
    file_dir: str
    name: str
    ext: str
    # Channel data:
    value: List[1]

    @classmethod
    def from_mat(cls,
                 file_dir: str,
                 name: str,
                 ext: str
                 ) -> RoughnessData:
        """
        Extracts the ChannelData from the `mat` file which lives inside
        `file_dir` under name `name` with extension `ext` (usually mat).
        """
        logger.debug(  # type: ignore
            f"\t Processing channel {name} in {file_dir} . . ."
        )
        # Extract the data from the `mat` file:
        ext = ext if '.' not in ext else ext.replace('.', '')
        mat_fname = os.path.join(file_dir, f'{name}.{ext}')
        mat_contents = scipy.io.loadmat(mat_fname)  # LOADING CODE
        core_data = mat_contents['all']

        # Load all roughness data from Zygo:
        
        # value: List[1]=[]
        value=torch.tensor(core_data)
        
        # for i in range(core_data.size):
        #     logger.debug(f"\t\t Processing section {i} . . .")
            
            # data_struct = core_data[i][0][0][0]
            # sections.append(ChannelSection.from_mat(
            #     statFx=data_struct['statFx'],
            #     statFy=data_struct['statFy'],
            #     statFz=data_struct['statFz'],
            #     Mic_FFT=data_struct['Mic_FFT'],
            #     AE_FFT=data_struct['AE_FFT'],
            #     Mic_FFT_BG=data_struct['Mic_FFT_BG'],
            #     AE_FFT_BG=data_struct['AE_FFT_BG']
            # ))

        # Extract the channel number:
        # channel_nums = re.findall(r'\d+', name)
        # if len(channel_nums) == 0:
        #     raise ValueError(
        #         "Malformed `mat` file channel name. "
        #         "Channel name should contain one and only one group of digits. "
        #         f"In file name `{name}.{ext}`, no groups of digits were found, "
        #         "so the channel number is unknown."
        #     )
        # if len(channel_nums) > 1:
        #     raise ValueError(
        #         "Malformed `mat` file channel name. "
        #         "Channel name should contain one and only one group of digits. "
        #         f"In file name `{name}.{ext}`, {len(channel_nums)} groups of "
        #         f"digits were found: {channel_nums} ."
        #     )
        # channel_num = int(channel_nums[0])

        return cls(
            file_dir=file_dir,
            name=name,
            ext=ext,
            value=value
        )




@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class ChannelData:
    """Container for file metadata and `ChannelSection` data derived from a file."""
    # File metadata:
    file_dir: str
    name: str
    ext: str
    # Channel data:
    channel_num: int
    sections: List[ChannelSection]

    @classmethod
    def from_mat(cls,
                 file_dir: str,
                 name: str,
                 ext: str
                 ) -> ChannelData:
        """
        Extracts the ChannelData from the `mat` file which lives inside
        `file_dir` under name `name` with extension `ext` (usually mat).
        """
        logger.debug(  # type: ignore
            f"\t Processing channel {name} in {file_dir} . . ."
        )
        # Extract the data from the `mat` file:
        ext = ext if '.' not in ext else ext.replace('.', '')
        mat_fname = os.path.join(file_dir, f'{name}.{ext}')
        mat_contents = scipy.io.loadmat(mat_fname)  # LOADING CODE
        core_data = mat_contents['LSTMinput']

        # Load all section data:
        sections: List[ChannelSection] = []
        for i in range(core_data.size):
            logger.debug(f"\t\t Processing section {i} . . .")
            data_struct = core_data[i][0][0][0]
            sections.append(ChannelSection.from_mat(
                
                statFxraw=data_struct['statFx'],
                statFyraw=data_struct['statFy'],
                statFzraw=data_struct['statFz'],
                
                statFxcomp=data_struct['statCompFx'],
                statFycomp=data_struct['statCompFy'],
                statFzcomp=data_struct['statCompFz'],
                
                Mic_FFT=data_struct['Mic_FFT'],
                AE_FFT=data_struct['AE_FFT'],
                Mic_FFT_BG=data_struct['Mic_FFT_BG'],
                AE_FFT_BG=data_struct['AE_FFT_BG'],
                
                Mic_SPEC=data_struct['Spect_Micm'],
                AE_SPEC=data_struct['Spect_AEm'],
                Mic_SPEC_BG=data_struct['Spect_Micbg'],
                AE_SPEC_BG=data_struct['Spect_AEbg']
                
            ))

        # Extract the channel number:
        channel_nums = re.findall(r'\d+', name)
        if len(channel_nums) == 0:
            raise ValueError(
                "Malformed `mat` file channel name. "
                "Channel name should contain one and only one group of digits. "
                f"In file name `{name}.{ext}`, no groups of digits were found, "
                "so the channel number is unknown."
            )
        if len(channel_nums) > 1:
            raise ValueError(
                "Malformed `mat` file channel name. "
                "Channel name should contain one and only one group of digits. "
                f"In file name `{name}.{ext}`, {len(channel_nums)} groups of "
                f"digits were found: {channel_nums} ."
            )
        channel_num = int(channel_nums[0])

        return cls(
            file_dir=file_dir,
            name=name,
            ext=ext,
            channel_num=channel_num,
            sections=sections
        )


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class ChannelSection:
    """
    Container for all processed and unprocessed data for each channel section.
    """
    force3D_mu: torch.Tensor
    force3D_sig: torch.Tensor
    
    statFxraw: torch.Tensor
    statFyraw: torch.Tensor
    statFzraw: torch.Tensor
    
    statFxcomp: torch.Tensor
    statFycomp: torch.Tensor
    statFzcomp: torch.Tensor
    
    Mic_FFT: torch.Tensor
    AE_FFT: torch.Tensor
    Mic_FFT_BG: torch.Tensor
    AE_FFT_BG: torch.Tensor
    
    Mic_SPEC: torch.Tensor
    AE_SPEC: torch.Tensor
    Mic_SPEC_BG: torch.Tensor
    AE_SPEC_BG: torch.Tensor    
    
    
    """
    # @property decorator; a pythonic way to use getters and setters in object-oriented programming.
    
    # Python programming provides us with a built-in @property decorator which makes usage of getter and setters much easier in Object-Oriented Programming.
    
    """
    # Useful accessors:    
    @property
    def F3D_mu(self) -> torch.Tensor: return self.force3D_mu
    @property
    def F3D_sig(self) -> torch.Tensor: return self.force3D_sig
    
    @property
    def Fxraw_mu(self) -> torch.Tensor: return self.statFxraw[0]
    @property
    def Fxraw_sig(self) -> torch.Tensor: return self.statFxraw[1]
    
    @property
    def Fyraw_mu(self) -> torch.Tensor: return self.statFyraw[0]
    @property
    def Fyraw_sig(self) -> torch.Tensor: return self.statFyraw[1]
    
    @property
    def Fzraw_mu(self) -> torch.Tensor: return self.statFzraw[0]
    @property
    def Fzraw_sig(self) -> torch.Tensor: return self.statFzraw[1]
    
    
    @property
    def Fxcomp_mu(self) -> torch.Tensor: return self.statFxcomp[0]
    @property
    def Fxcomp_sig(self) -> torch.Tensor: return self.statFxcomp[1]
    
    @property
    def Fycomp_mu(self) -> torch.Tensor: return self.statFycomp[0]
    @property
    def Fycomp_sig(self) -> torch.Tensor: return self.statFycomp[1]
    
    @property
    def Fzcomp_mu(self) -> torch.Tensor: return self.statFzcomp[0]
    @property
    def Fzcomp_sig(self) -> torch.Tensor: return self.statFzcomp[1]
    
    
    @property
    def Mic_FFT_val(self) -> torch.Tensor: return self.Mic_FFT[0]
    @property
    def Mic_FFT_freq(self) -> torch.Tensor: return self.Mic_FFT[1]
    
    @property
    def AE_FFT_val(self) -> torch.Tensor: return self.AE_FFT[0]
    @property
    def AE_FFT_freq(self) -> torch.Tensor: return self.AE_FFT[1]
    
    @property
    def Mic_FFT_BG_val(self) -> torch.Tensor: return self.Mic_FFT_BG[0]
    @property
    def Mic_FFT_BG_freq(self) -> torch.Tensor: return self.Mic_FFT_BG[1]
    
    @property
    def AE_FFT_BG_val(self) -> torch.Tensor: return self.AE_FFT_BG[0]
    @property
    def AE_FFT_BG_freq(self) -> torch.Tensor: return self.AE_FFT_BG[1]
    
    
    @property
    def Mic_SPEC(self) -> torch.Tensor: return self.Mic_SPEC
    @property
    def AE_SPEC(self) -> torch.Tensor: return self.AE_SPEC
    
    @property
    def Mic_SPEC_BG(self) -> torch.Tensor: return self.Mic_SPEC_BG
    @property
    def AE_SPEC_BG(self) -> torch.Tensor: return self.AE_SPEC_BG
    
    
    @classmethod
    def from_mat(cls,
                 statFxraw: np.ndarray,
                 statFyraw: np.ndarray,
                 statFzraw: np.ndarray,
                 
                 statFxcomp: np.ndarray,
                 statFycomp: np.ndarray,
                 statFzcomp: np.ndarray,
                 
                 Mic_FFT: np.ndarray,
                 AE_FFT: np.ndarray,
                 Mic_FFT_BG: np.ndarray,
                 AE_FFT_BG: np.ndarray,
                 
                 Mic_SPEC:np.ndarray,
                 AE_SPEC:np.ndarray,
                 Mic_SPEC_BG:np.ndarray,
                 AE_SPEC_BG:np.ndarray,
                 
                 
                 force_target_size: int = 180  # target size for force vectors
                 ) -> ChannelSection:
        """
        Creates a `ChannelSection` from the data extracted from a `mat` file.
        """
        # Upsample all force data:
        statFx_up = np.zeros((2, force_target_size))
        statFy_up = np.zeros((2, force_target_size))
        statFz_up = np.zeros((2, force_target_size))
        for f in [(statFxcomp, statFx_up), (statFycomp, statFy_up), (statFzcomp, statFz_up)]:
            # For all stat channel (mu, sigma):
            for i in range(2):
                f[1][i, :] = resample_vector(
                    f[0][i, :], force_target_size, kind='linear'
                )
    
        # Build 3D Force Structures:
        force3D_mu = torch.stack((
            torch.tensor(statFx_up[0, :]),
            torch.tensor(statFy_up[0, :]),
            torch.tensor(statFz_up[0, :])
        ))
    
        force3D_sig = torch.stack((
            torch.tensor(statFx_up[1, :]),
            torch.tensor(statFy_up[1, :]),
            torch.tensor(statFz_up[1, :])
        ))
    
        # Subtract BG Noise as PSD:
        # ! TODO: Create spectrograms and diff.
    
        # Build and return structure:
        return cls(
            
            statFxraw=torch.tensor(statFxraw),
            statFyraw=torch.tensor(statFyraw),
            statFzraw=torch.tensor(statFzraw),
            
            force3D_mu=force3D_mu,
            force3D_sig=force3D_sig,
            statFxcomp=torch.tensor(statFx_up),
            statFycomp=torch.tensor(statFy_up),
            statFzcomp=torch.tensor(statFz_up),
            Mic_FFT=torch.tensor(Mic_FFT),
            AE_FFT=torch.tensor(AE_FFT),
            Mic_FFT_BG=torch.tensor(Mic_FFT_BG),
            AE_FFT_BG=torch.tensor(AE_FFT_BG),
            
            Mic_SPEC=torch.tensor(Mic_SPEC),
            AE_SPEC=torch.tensor(AE_SPEC),
            Mic_SPEC_BG=torch.tensor(Mic_SPEC_BG),
            AE_SPEC_BG=torch.tensor(AE_SPEC_BG)
            
        )
