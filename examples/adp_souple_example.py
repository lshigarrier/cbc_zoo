import torch
import cbc_zoo as cbc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adpsouple = cbc.get_cbc_model('ADPSouple', device, verbose=True)


if __name__ == '__main__':
    main()
