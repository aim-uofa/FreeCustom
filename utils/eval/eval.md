The format of the data is:

```
.
├── multi-concept
│   ├── input
│   │   ├── dog_flower
│   │   │   ├── image
│   │   │   │   ├── 0.jpg
│   │   │   │   └── 1.jpg
│   │   │   ├── mask
│   │   │   │   ├── 0.jpg
│   │   │   │   └── 1.jpg
│   │   │   └── prompt.txt
│   │   ├── hepburn_sunglasses_beard
│   │   │   ├── image
...
│   ├── cd
│   │   ├── dog_flower
│   │   │   ├── 1.jpg
│   │   │   ├── ...
│   │   │   └── 5.jpg
│   │   ├── hepburn_sunglasses_beard
│   │   │   ├── 1.jpg
...
│   ├── ours
│   │   ├── dog_flower
│   │   │   ├── 1.jpg
...
│   └── perfusion
│   │   ├── dog_flower
│   │   │   ├── 1.jpg
...
└── single-concept
    ├──  input
    │   ├── cat0.png
    │   ├── cat1.png
    │   ├── dog0.png
    │   ├── dog2.png
    │   └── Thanos.png
    ├── ours
    │   ├── cat0 "a photo of a cat in a jungle" seed_1024.png
    │   ├── ...
    │   └── Thanos "a photo of Thanos with the Effiel Tower in the background" seed_970411.png
    ├── dreambooth
    │   ├── cat0 "a photo of a cat in a jungle" seed_1024.png
    ...
    ├── elite
    ...
    ├── neti
    ...
    └── blip_diffusion
        └──...
```