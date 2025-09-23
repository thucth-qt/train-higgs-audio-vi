#!/bin/bash
export 
export HIGGS_DTYPE="bfloat16"
# Set the Python script name
PYTHON_SCRIPT="trainer/merger.py" # Make sure this path points to your Python script

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at '$PYTHON_SCRIPT'. Please ensure it's in the correct directory or update the script path."
    exit 1
fi

# Default parameters for the Vietnamese pipeline
BASE_MODEL_PATH="/root/data/higgs/weights/higgs-audio-v2-generation-3B-base"
LORA_ADAPTER_PATH="/root/data/higgs/train-higgs-audio-vi/runs/2_output_vn_lora_mini/lora_adapters"
OUTPUT_PATH="/root/data/higgs/train-higgs-audio-vi/runs/2_output_vn_lora_mini_merged"
COMPARE_MODELS_FLAG=""
TEST_INPUT_VALUE=""
SAVE_TOKENIZER_FLAG="" # Default is to save the tokenizer

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift
            ;;
        --lora_adapter_path)
            LORA_ADAPTER_PATH="$2"
            shift
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift
            ;;
        --compare_models)
            COMPARE_MODELS_FLAG="--compare_models"
            ;;
        --test_input)
            TEST_INPUT_VALUE="--test_input \"$2\"" # Ensure spaces in the string are handled correctly
            shift
            ;;
        --no_save_tokenizer) # If the user specifies not to save the tokenizer
            SAVE_TOKENIZER_FLAG="--no_save_tokenizer" # pass this flag to the Python script
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Check required parameters
if [ -z "$BASE_MODEL_PATH" ]; then
    echo "Error: --base_model_path must be specified."
    echo "Usage: ./merge_model.sh --base_model_path <path_to_base_model> --lora_adapter_path <path_to_lora> --output_path <path_to_output> [optional_args]"
    exit 1
fi

if [ -z "$LORA_ADAPTER_PATH" ]; then
    echo "Error: --lora_adapter_path must be specified."
    echo "Usage: ./merge_model.sh --base_model_path <path_to_base_model> --lora_adapter_path <path_to_lora> --output_path <path_to_output> [optional_args]"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output_path must be specified."
    echo "Usage: ./merge_model.sh --base_model_path <path_to_base_model> --lora_adapter_path <path_to_lora> --output_path <path_to_output> [optional_args]"
    exit 1
fi

# Print the command to be executed
echo "Executing LoRA model merge..."
echo "Base model path: $BASE_MODEL_PATH"
echo "LoRA adapter path: $LORA_ADAPTER_PATH"
echo "Output path: $OUTPUT_PATH"
[ -n "$COMPARE_MODELS_FLAG" ] && echo "Will perform model comparison"
[ -n "$TEST_INPUT_VALUE" ] && echo "Test input: $(echo $TEST_INPUT_VALUE | cut -d\" -f2)"
if [ -z "$SAVE_TOKENIZER_FLAG" ]; then # If SAVE_TOKENIZER_FLAG is empty, it means save by default
    echo "Will save tokenizer (default behavior)"
else
    echo "Will not save tokenizer ($SAVE_TOKENIZER_FLAG)"
fi


# Build and execute the Python command
CMD="python3 $PYTHON_SCRIPT \
    --base_model_path \"$BASE_MODEL_PATH\" \
    --lora_adapter_path \"$LORA_ADAPTER_PATH\" \
    --output_path \"$OUTPUT_PATH\" \
    $COMPARE_MODELS_FLAG \
    $TEST_INPUT_VALUE \
    $SAVE_TOKENIZER_FLAG" # Directly pass SAVE_TOKENIZER_FLAG, which is either "--no_save_tokenizer" or an empty string

echo "Command: $CMD"
eval $CMD

# Check command execution result
if [ $? -eq 0 ]; then
    echo "LoRA merge process completed successfully!"
else
    echo "LoRA merge process failed, please check the error messages."
    exit 1
fi