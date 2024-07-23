#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:19:40 2024

@author: joao
"""

import torch
import esm
import random

# Load ESM-2 model
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

def insert_mask(sequence, position, mask="<mask>"):
    """
    Replaces a character in a given position of a sequence with a mask.

    Parameters:
    - sequence (str or list): The sequence to replace the character in.
    - position (int): The position in the sequence where the character should be replaced.
    - mask (str): The mask to insert (default is "<mask>").

    Returns:
    - str or list: The sequence with the mask replacing the character at the specified position.
    """
    
    if not (0 <= position < len(sequence)):
        raise ValueError("Position is out of bounds.")
    
    if isinstance(sequence, str):
        return sequence[:position] + mask + sequence[position + 1:]
    elif isinstance(sequence, list):
        return sequence[:position] + [mask] + sequence[position + 1:]
    else:
        raise TypeError("Sequence must be a string or list.")

def complete_mask(input_sequence, posi, temperature=1.0):
    
    data = [
        ("protein1", insert_mask(input_sequence, posi, mask="<mask>"))
    ]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Predict masked tokens
    with torch.no_grad():
        token_probs = model(batch_tokens, repr_layers=[33])["logits"]

    # Apply temperature
    token_probs /= temperature
    
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(token_probs)

    # Get the index of the <mask> token
    mask_idx = (batch_tokens == alphabet.mask_idx).nonzero(as_tuple=True)

    # Extract the predicted tokens for the masked positions
    predicted_tokens = torch.argmax(probabilities, dim=-1)
    
    # Sample from the probability distribution
    predicted_tokens = torch.multinomial(probabilities[mask_idx], num_samples=1).squeeze(-1)

    # Replace the <mask> token with the predicted token
    batch_tokens[mask_idx] = predicted_tokens


    predicted_residues = [alphabet.get_tok(pred.item()) for pred in batch_tokens[0]]
    
    seq_predicted = ''.join(predicted_residues[1:-1])
    
    if input_sequence != seq_predicted:
        print("Mutation added!! 😉")
    
    return seq_predicted
    
def create_masked_sequences(sequence, masked_pos, cdrs_or_fm):
    
    if cdrs_or_fm == "cdr":
        masked_sequences = []
        for i in range(len(sequence)):
            if masked_pos[i] == 1:
                masked_sequence = sequence[:i] + "<mask>" + sequence[i+1:]
                masked_sequences.append((f"protein1 with mask at position {i+1}", masked_sequence))
        return masked_sequences
    if cdrs_or_fm == "fm":
        masked_sequences = []
        for i in range(len(sequence)):
            if masked_pos[i] == 0:
                masked_sequence = sequence[:i] + "<mask>" + sequence[i+1:]
                masked_sequences.append((f"protein1 with mask at position {i+1}", masked_sequence))
        return masked_sequences


def generate_Sequence(input_sequence, cdrs, loc, temperature = 1.0, order='random'):
    
    new_sequence_temp = input_sequence
    
    indices = list(range(len(input_sequence)))
    
    if loc == "all":
        if order == 'random':
            random.shuffle(indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            indices.reverse()
        for i in indices:
            print(i)
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)
            
    elif loc == "cdr":
        cdr_indices = [i for i in indices if i in cdrs]
        if order == 'random':
            random.shuffle(cdr_indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            cdr_indices.reverse()
        for i in cdr_indices:
            print(i)
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)
            
    elif loc == "fm":
        fm_indices = [i for i in indices if i not in cdrs]
        if order == 'random':
            random.shuffle(fm_indices)
        elif order == 'forward':
            pass  # already in forward order
        elif order == 'backward':
            fm_indices.reverse()
        for i in fm_indices:
            new_sequence_temp = complete_mask(input_sequence=new_sequence_temp, posi=i, temperature=temperature)
    
    return new_sequence_temp

def generate_batch(total_sequences, input_sequence, cdrs, loc, temperature = 1.0, order='random'):
    
    new_sequences = []
    
    for i in range(0, total_sequences):
        temp_seq = generate_Sequence(input_sequence, cdrs, loc, temperature, order)
        new_sequences.append(temp_seq)
    
        
    with open(f"VHH_esm2_{loc}_{order}_{temperature}.fasta", "w") as file:
        for i in range(0, total_sequences):
            file.write(f">seq{i}\n{new_sequences[i]}\n")


input_sequence = "MADVQLQASGGGLVQAGGSLRLSCAASGNINTIDVMGWYRQAPGKQRELVADITRLASANYADSVKGRFTISRDNAKNTVYLQMNNLEPKDTAVYYCAQWILSTDHSYMHYWGQGTQVTVTVSS"

cdrs = [25, 26, 27, 28, 29, 30, 31, 51, 52, 53, 54, 55, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

#### Generate for all positions
all_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="all")
#### Generate for CDRs
CDRs_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="cdr", order = "random") 
#### Generate for frameworks
FM_new = generate_Sequence(input_sequence=input_sequence, cdrs=cdrs, loc="fm")



#### Generate a batch with 5 sequences
generate_batch(5, input_sequence, cdrs, "cdr", 1, "random")


cetuximab = "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENKCKATGQDILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPSRFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRQVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSAASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#171 a 601

CDRs = [412, 413, 414, 415, 416, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 220, 221, 222, 223, 224, 225, 226, 259, 260, 261, 262, 263, 264, 265, 266, 267]

CDRs_new = [index - 171 for index in CDRs]

only_fab = cetuximab[171:]


generate_batch(50, only_fab, CDRs_new, "cdr", 1.2, "random")

##### 

petase_wt = "MQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEHHHHHH"

locked_pos = [0, 58, 59, 60, 61, 62, 63, 66, 89, 90, 91, 92, 93, 131, 132, 133, 134, 135, 136, 137, 138, 155, 156, 157, 158, 159, 160, 176, 179, 180, 181, 182, 183, 186, 187, 210, 211, 212, 214, 246, 262, 266, 267, 268, 269, 270, 271]

generate_batch(50, petase_wt, locked_pos, "fm", 1.2, "random")

petase_wt[locked_pos[0]]
