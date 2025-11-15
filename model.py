from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor
import yaml
import os
from peft import LoraConfig, get_peft_model



def load_model():
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'qlora_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['bnb_config']['load_in_4bit'],
        bnb_4bit_use_double_quant=config['bnb_config']['bnb_4bit_use_double_quant'],
        bnb4bit_quant_type=config['bnb_config']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=config['bnb_config']['bnb_4bit_compute_dtype'],  # Convert from string to torch dtype
        llm_int8_enable_fp32_cpu_offload=config['bnb_config']['llm_int8_enable_fp32_cpu_offload']
    )

    model = Blip2ForConditionalGeneration.from_pretrained(config['model_config']['model_name'],
                                    device_map=config['model_config']['device_map'],
                                    quantization_config=bnb_config,
                                    trust_remote_code=config['model_config']['trust_remote_code'],
                                    low_cpu_mem_usage=config['model_config']['low_cpu_mem_usage'])


    processor = Blip2Processor.from_pretrained(config['processor_config']['name'])


    target_modules = config['qformer_target_modules']

    # Let's define the LoraConfig
    config = LoraConfig(
        r=config['lora_config']['r'],
        lora_alpha=config['lora_config']['lora_alpha'],
        lora_dropout=config['lora_config']['lora_dropout'],
        bias=config['lora_config']['bias'],
        target_modules=target_modules
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model, processor

if __name__ == "__main__":
    model, processor = load_model()
    print("Model and processor loaded successfully.")