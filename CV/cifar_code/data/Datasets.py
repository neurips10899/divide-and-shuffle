from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
#from data.Caltech101_dataset import Caltech101
#
import os

def get_Transforms(args):
    
    transform= None
    dataset = args.dataset.strip().upper()
    
    if dataset == 'CALTECH101':
        # Image transformations        
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
    elif dataset == 'IMAGENET':
        # Image transformations        
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
    elif dataset == 'CIFAR10':
        # Image transformations        
        transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])  # Imagenet standards
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], 
                                     [0.5, 0.5, 0.5])
            ]),
        }    
    elif dataset == 'CIFAR100':
        # Image transformations        
        transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
    elif dataset == 'MNIST':
        
        transform = {'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'test':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])   
    }
    
    return transform


def get_Dataloader(args):
    
    transform = get_Transforms(args)
    dataset=args.dataset.strip().upper()
    train_batch_size=args.batch_size
    test_batch_size=args.batch_size
    num_workers = args.num_workers
    
    data_path=args.data_path

    if dataset == 'MNIST':
        train_set = datasets.MNIST(data_path, train=True, transform=transform['train'], download=True)
        test_set = datasets.MNIST(data_path, train=False, transform=transform['test'], download=True)
    elif dataset =='CIFAR10':
        train_set = datasets.CIFAR10(data_path, train=True, transform=transform['train'], download=True)
        test_set = datasets.CIFAR10(data_path, train=False, transform=transform['test'], download=True)
    elif dataset =='CIFAR100':
        train_set = datasets.CIFAR100(data_path, train= True, transform=transform['train'], download=True)
        test_set = datasets.CIFAR100(data_path, train= False, transform=transform['test'], download=True)
    elif dataset =='IMAGENET':
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
        train_set = datasets.ImageFolder(traindir, transform=transform['train'] )
        test_set = datasets.ImageFolder(valdir, transform=transform['test'])
    elif dataset == 'CALTECH101':
        train_set= Caltech101(data_path,True,transform=transform['train'],download=True)
        test_set= Caltech101(data_path,False,transform=transform['test'],download=True)
    else:
        assert 1==0, 'invalid dataset'
    
    sampler = DistributedSampler(train_set)
    train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=(sampler is None),num_workers=num_workers, sampler=sampler,timeout=3000)
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,num_workers=num_workers, timeout=3000)

    return train_loader, test_loader









