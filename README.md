# ESM2_NanoGEN

This project uses the ESM2 language model (https://github.com/facebookresearch/esm) to generate sequences for antibody and antibody fragments. The primary objective is to generate antibody sequences with specific Complementarity-Determining Regions (CDRs) and framework modifications.

<div style="display: flex; justify-content: flex-start;">
  <img src="https://media.giphy.com/media/OAyIxZwcW1ZwKW8V2P/giphy.gif" height="300" width="400" style="margin-right: 10px;" />
  <img src="https://media.giphy.com/media/Q7z1lwDHFZBOASkIDD/giphy.gif" height="300" width="400" />
</div>

## Installation
Setting up the environment using the .yaml:
<br />

1. Clone the repository:
    ```sh
    git clone https://github.com/jsartori12/ESM2_NanoGEN.git
    cd ESM2_NanoGEN
    ```
2. Download ESM2 repository:
    ```sh
    git clone https://github.com/facebookresearch/esm.git
    ```
3. Move esm2 content to the repository directory:
    ```sh
    mv esm/* ESM2_NanoGEN
    ```
4. Create the environment:
    ```sh
    conda env create -f environment.yaml
    ```
5. Activate the environment:
    ```sh
    conda activate esmfold
    ```

## Running the code

### Parameters

  | Parameter | Description  | 
  | :---:   | :---: |
  |total_sequences | Number of sequences to generate
  |input_sequence | Input sequence to use as template.
  |cdrs | List containing the CDRs positions
  |loc | Regions to design. "cdrs", "fm" for framework and "all"
  |temperature | Temperature applied to logits in token prediction step
  |order | Starting point for the predictions. "foward", "backward" or "random"
  

### Designing Options
functions_esm2.py contain 3 usage examples for generate from the entire sequence, only for CDRs and only for frameworks

<br />

```python
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

```

