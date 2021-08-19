def train():
 transform = transforms.Compose(
 [
 transforms.Resize((356, 356)),
 transforms.RandomCrop((299, 299)),
 transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 ]
 )
 train_loader, dataset = get_loader(
 root_folder="flickr8k/images",
 annotation_file="flickr8k/captions.txt",
 transform=transform,
 num_workers=2,
 )
 torch.backends.cudnn.benchmark = True
 # device = torch.device("cpu")
 device = torch.device("cuda")
 load_model = True
 save_model = True
 train_CNN = False
 torch.cuda.empty_cache()
 # Hyperparameters
 embed_size = 256
 hidden_size = 256
 vocab_size = len(dataset.vocab)
 num_layers = 1
 learning_rate = 3e-4
 num_epochs = 100
 # initialize model, loss etc
 model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers
).to(device)
 criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi[
"<PAD>"])
 optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 # Only finetune the CNN
 for name, param in model.encoderCNN.inception.named_parameters()
:
 if "fc.weight" in name or "fc.bias" in name:
 param.requires_grad = True
 else:
 param.requires_grad = train_CNN
 if load_model:
 step = load_checkpoint(torch.load("my_checkpoint.pth.tar"),
model, optimizer)
 model.train()
 for epoch in range(num_epochs):
 # Uncomment the line below to see a couple of test cases
 print_examples(model, device, dataset)
 if save_model:
 checkpoint = {
 "state_dict": model.state_dict(),
 "optimizer": optimizer.state_dict(),
 "step": step,
 }
 save_checkpoint(checkpoint)
 for idx, (imgs, captions) in tqdm(
 enumerate(train_loader), total=len(train_loader), leave=
False
 ):
 imgs = imgs.to(device)
 captions = captions.to(device)
 outputs = model(imgs, captions[:-1])
 loss = criterion(
 outputs.reshape(-
1, outputs.shape[2]), captions.reshape(-1)
 )
