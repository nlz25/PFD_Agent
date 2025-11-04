import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from dotenv import load_dotenv
from ase import Atoms
from ase.io import write,read
import numpy as np
import random


import logging

def _h_filter_cpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    """Filter configurations based on entropy.

    Args:
        iter_confs (List[Atoms]): The configurations to iterate over.
        dset_confs (List[Atoms], optional): The reference configurations. Defaults to [].
        chunk_size (int, optional): The number of configurations to process at once. Defaults to 10.
        max_sel (int, optional): _description_. Defaults to 100.
        k (int, optional): _description_. Defaults to 32.
        cutoff (float, optional): _description_. Defaults to 5.0.
        batch_size (int, optional): _description_. Defaults to 1000.
        h (float, optional): _description_. Defaults to 0.015.
        dtype (str, optional): _description_. Defaults to 'float32'.

    Returns:
        _type_: _description_
    """
    from quests.descriptor import get_descriptors
    from quests.entropy import entropy,delta_entropy
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size
        
    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype)
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)
    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end
        
    H_list = []
    # initial entropy
    H= entropy(dset_desc, h=h, batch_size=batch_size)
    logging.info(f"Initial entropy with {len(dset_confs)} reference configurations: {H:.4f}")
    H_list.append(H)
    result = {}
    result.update({"iter_00": H, "num_confs": len(dset_confs)})
    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = np.vstack(re_desc)
        delta = delta_entropy(x, dset_desc, h=h,batch_size=batch_size)
        num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
        atom_indices = []
        start = 0
        for n in num_atoms_per_structure:
            end = start + n
            atom_indices.append((start, end))
            start = end
        delta_sums = [delta[start:end].sum() for start, end in atom_indices]
        sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
        sorted_re_indices = [idx for idx, _ in sorted_pairs]
        selected_indices = sorted_re_indices[:chunk_size]
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        H = entropy(dset_desc, h=h, batch_size=batch_size)
        dH = H - H_list[-1]
        H_list.append(H)
        logging.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
        result.update({f"iter_{ii+1:02d}": H, "num_confs": len(dset_confs)})
        if dH < 1e-2:
            logging.info(f"Entropy increase {dH:.4f} is less than 1e-2, stopping selection.")
            break
    return dset_confs[num_ref:], result # return only the newly selected ones

def _h_filter_gpu(
    iter_confs: List[Atoms],
    dset_confs: List[Atoms]=[],
    chunk_size: int = 10,
    max_sel: int = 100,
    k=32,
    cutoff=5.0,
    batch_size: int = 1000,
    h = 0.015,
    dtype='float32',
    **kwargs
):
    import torch
    from quests.descriptor import get_descriptors
    from quests.gpu.entropy import delta_entropy,entropy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # logging results for filtering
    result = {}
    num_ref=len(dset_confs)
    if len(dset_confs) == 0:
        if chunk_size >= len(iter_confs):
            return iter_confs
        random.shuffle(iter_confs)
        dset_confs = iter_confs[:chunk_size]
        iter_confs = iter_confs[chunk_size:]
        num_ref=0
        max_sel-= chunk_size

    max_iter = min(max_sel//chunk_size+(max_sel%chunk_size>0), 
                   len(iter_confs)//chunk_size+(len(iter_confs)%chunk_size>0))
    
    iter_desc = get_descriptors(iter_confs, k=k, cutoff=cutoff,dtype=dtype)
    dset_desc = get_descriptors(dset_confs, k=k, cutoff=cutoff,dtype=dtype)

    num_atoms_per_structure_iter = [atoms.get_number_of_atoms() for atoms in iter_confs]
    atom_indices_iter = []
    start = 0
    for n in num_atoms_per_structure_iter:
        end = start + n
        atom_indices_iter.append((start, end))
        start = end
    
    H_list = []
    x= torch.tensor(dset_desc,device=device, dtype=torch.float32)
    H= entropy(x, h=h, batch_size=batch_size,device=device)
    logging.info(f"Initial entropy with {len(dset_confs)} reference configurations: {H:.4f}")
    H_list.append(float(H.cpu().numpy()))
    result.update({"num_confs": len(dset_confs),"iter_00": float(H.cpu().numpy())})
    indices = []
    for ii in range(max_iter):
        re_indices = [i for i in range(len(iter_confs)) if i not in indices]
        re_confs = [iter_confs[i] for i in re_indices]
        re_desc = [iter_desc[atom_indices_iter[i][0]:atom_indices_iter[i][1]] for i in re_indices]
        x = torch.tensor(np.vstack(re_desc),device=device, dtype=torch.float32)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        delta = delta_entropy(x, y, h=h,batch_size=batch_size, device=device)
        delta = delta.cpu().numpy()
        num_atoms_per_structure = [atoms.get_number_of_atoms() for atoms in re_confs]
        atom_indices = []
        start = 0
        for n in num_atoms_per_structure:
            end = start + n
            atom_indices.append((start, end))
            start = end
        delta_sums = [delta[start:end].sum() for start, end in atom_indices]
        sorted_pairs = sorted(zip(re_indices, delta_sums), key=lambda x: x[1], reverse=True)
        sorted_re_indices = [idx for idx, _ in sorted_pairs]
        selected_indices = sorted_re_indices[:chunk_size]
        dset_desc_ls=[dset_desc]
        for idx in selected_indices:
            indices.append(idx)
            dset_confs.append(iter_confs[idx])
            dset_desc_ls.append(iter_desc[atom_indices_iter[idx][0]:atom_indices_iter[idx][1]])
        dset_desc = np.vstack(dset_desc_ls)
        y = torch.tensor(dset_desc, device=device, dtype=torch.float32)
        H = entropy(y, h=h, batch_size=batch_size,device=device)
        dH = H - H_list[-1]
        H_list.append(float(H.cpu().numpy()))
        logging.info(f"Iteration {ii+1}/{max_iter}, selected {len(dset_confs)} configurations, entropy {H:.4f}")
        result.update({f"iter_{ii+1:02d}": float(H.cpu().numpy()), "num_confs": len(dset_confs)})
        if dH < 1e-2:
            logging.info(f"Entropy increase {float(dH.cpu().numpy()):.4f} is less than 1e-2, stopping selection.")
            break
    return dset_confs[num_ref:], result # return only the newly selected ones