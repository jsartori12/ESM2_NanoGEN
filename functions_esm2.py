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
    
    
    exclude = [alphabet.get_idx('X'), alphabet.get_idx('B'),
               alphabet.get_idx('U'), alphabet.get_idx('Z'),
               alphabet.get_idx('O')]
    
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
    
    # Zero out probabilities for excluded tokens
    for token in exclude:
        probabilities[:, :, token] = 0.0

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


