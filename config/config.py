from pathlib import Path

root_path=Path(__file__).resolve().parent.parent

data_for_ml=root_path/'Data for ml'

valid_data=root_path/'Valid_Data'



if __name__=='__main__':
    print(root_path)