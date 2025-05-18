import torch
import unicodedata
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def is_ascii(str):
    for char in str:
        if ord(char) > 128:
            return False
    return True



class NamesDataset(Dataset):
    
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class Data:
    
    def __init__(self):
        self.data = None
        self.vocab = None
        self.dataset = None
        self.dataLoader = None
        self.categories = None
        
        self._read()
        # self._generateLoader()
        
        
    def _flatten(self, data):
        self.categories = categories = sorted(list(data.keys()))
        
        # flattened = []
        mapping = {}
        for key, value in data.items():
            values = []
            for item in value:
                values.append((self.word2tensor(item), categories.index(key)))
            mapping[key] = values
        return mapping
    
    def _read(self):
        root_dir = Path(__file__).parent


        data_dir = Path(f"{root_dir}/data")

        vocab = set()
        data = {}
        for i, file_path in enumerate(data_dir.iterdir()):
            # if i == 2:
                # break
            if file_path.is_file():

                file_name = file_path.name.split('.')[0]
                print(file_name)

                with open(file_path, 'r', encoding='utf-8') as f:
                    
                    texts = f.readlines()
                    
                    data[file_name] = []
                    for text in texts:
                        if not is_ascii(text):
                            continue
                        cleaned = remove_accents(text)
                        data[file_name].append(cleaned)
                        vocab.update(char for char in cleaned)

        self.vocab = sorted(vocab)
        self.categories = sorted(list(data.keys()))
        self.data = data
        return self.data, self.vocab
    
    
    def random_sample(self):
        random_category = np.random.choice(self.categories, replace=True)
        random_word = np.random.choice(self.data[random_category], replace=True)
        
        return self.word2tensor(random_word), torch.tensor([self.categories.index(random_category)], dtype=torch.long)
    
    def _generateLoader(self):
        self.dataset, self.dataLoader = NamesDataset(self.data), DataLoader(self.data, batch_size=1, sampler=RandomSampler(self.data, replacement=True, num_samples=1))

    def word2tensor(self, word: str):
        encodings = []
        for char in word:
            encoding = torch.zeros(1, len(self.vocab))
            encoding[0][self.vocab.index(char)] = 1
            encodings.append(encoding)

        encodings = torch.stack(encodings)
        return encodings

    def tensor2word(self, tensor: torch.Tensor):
        word = ""
        for encoding in tensor:
            char = self.vocab[torch.argmax(encoding)]
            word += char
        return word
    
    def tensor2category(self, categoryIdx):
        return self.categories[categoryIdx]

    
        
        


if __name__ == '__main__':
    d = Data()
    w, c = d.random_sample()
    print(d.tensor2word(w), c)

    w, c = d.random_sample()
    print(d.tensor2word(w), c)

    w, c = d.random_sample()
    print(d.tensor2word(w), c)
