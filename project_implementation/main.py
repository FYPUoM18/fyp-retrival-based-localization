
import config
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator


steps=config.steps

if __name__ == "__main__":
    
    if 1 in steps:
        csv_generator=CSV_Generator(config)
        csv_generator.generate()

    if 2 in steps:
        data_compiler=Compiler(config)
        data_compiler.compile()