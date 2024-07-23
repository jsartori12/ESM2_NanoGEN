#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:40:03 2024

@author: joao
"""

from functions_esm2 import generate_Sequence
import torch
import esm

# Load ESM-2 model
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


input_sequence = "MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASANYADSVKGRFTISRDNAKNTVYLQMNNLEPKDTAVYYCAQWILSTDHSYMHYWGQGTQVTVTVSS"

cdrs = [25, 26, 27, 28, 29, 30, 31, 51, 52, 53, 54, 55, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

#### Generate for all positions
all_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="all")
#### Generate for CDRs
CDRs_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="cdr")
CDRs_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="cdr", order = "random") 

#### Generate for frameworks
FM_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="fm")


CDRs_new == input_sequence
