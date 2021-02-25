import torchimport torchvisionfrom torch import nnfrom torch.autograd import Variablefrom torch.utils.data import DataLoaderimport torch.utils.data as Datafrom torch.autograd import Functionfrom torchvision import transformsimport torchvision.transforms.functional as TFfrom torchvision.utils import save_imagefrom torchvision.datasets import MNISTimport osimport utilimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport randomimport cv2from numpy.fft import rfft2, irfft2use_cuda = torch.cuda.is_available()device = torch.device('cuda:0' if use_cuda else 'cpu')if not os.path.exists('./dc_img'):    os.mkdir('./dc_img')num_epochs = 10000batch_size = 128learning_rate = 1e-3class FaceLandmarksDataset(Data.Dataset):    """Face Landmarks dataset."""    def __init__(self, root_dir, csv_file, transform=None):        """        Args:            csv_file (string): Path to the csv file with annotations.            root_dir (string): Directory with all the images.            transform (callable, optional): Optional transform to be applied                on a sample.        """        util.read_bins_toCSV(root_dir, csv_file, 200, 200, True, True)        self.landmarks_frame = pd.read_csv(csv_file)        self.root_dir = root_dir    def __len__(self):        return len(self.landmarks_frame)    def transform(self, image, label):        # # Resize        # resize = transforms.Resize(size=(520, 520))        # image = resize(image)        # label = resize(label)        #        # # Random crop        # i, j, h, w = transforms.RandomCrop.get_params(        #     image, output_size=(512, 512))        # image = TF.crop(image, i, j, h, w)        # label = TF.crop(label, i, j, h, w)        image = TF.to_pil_image(image)        label = TF.to_pil_image(label)        # Random horizontal flipping        if random.random() > 0.5:            image = TF.hflip(image)            label = TF.hflip(label)        # Random vertical flipping        if random.random() > 0.5:            image = TF.vflip(image)            label = TF.vflip(label)        # Transform to tensor        image = TF.to_tensor(image)        label = TF.to_tensor(label)        # image = TF.normalize(image, mean=(0,), std=(1,))        # label = TF.normalize(label, mean=(0,), std=(1,))        return image, label    def __getitem__(self, idx):        if torch.is_tensor(idx):            idx = idx.tolist()        ipp = util.read_8bit_bin(self.landmarks_frame.iloc[idx, 2], (200, 200), True)        ipp = ((ipp - np.mean(ipp)) / np.std(ipp)).astype('float32')        label = np.zeros((ipp.shape[0], ipp.shape[1])).astype('float32')        ipp, label = self.transform(ipp, label)        return ipp, labeldataset = FaceLandmarksDataset(root_dir='D:\\data\\partial\\A52\\20201030_Partial_4DB\\11_P',                                    csv_file='list.csv')dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)class autoencoder(nn.Module):    def __init__(self):        super(autoencoder, self).__init__()        self.encoder = nn.Sequential(         # input shape (1, 200, 200)            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=200, kernel_size=5, stride=1, padding=2),            nn.ELU(),        )        self.decoder = nn.Sequential(            nn.Conv2d(in_channels=200, out_channels=64, kernel_size=5, stride=1, padding=2),            nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),            nn.ELU(),        )    def forward(self, x):        x = self.encoder(x)        x = self.decoder(x)        return xmodel = autoencoder().cuda()criterion = nn.MSELoss()optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,                             weight_decay=1e-5)# # and a learning rate scheduler# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,#                                                    step_size=1000,#                                                    gamma=0.01)N_TEST_IMG = 5f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))plt.ion()   # continuously plotfor epoch in range(num_epochs):    total_loss = 0    for step, (img, label) in enumerate(dataloader):        img = Variable(img).to(device=device)        label = Variable(label).to(device=device)        # ===================forward=====================        output = model(img)        loss = criterion(output, label)        total_loss += loss.data        # ===================backward====================        optimizer.zero_grad()        loss.backward()        optimizer.step()        if epoch % 10 == 0 and step == 0:            # plotting decoded image            for i in range(N_TEST_IMG):                a[0][i].clear()                a[0][i].imshow(np.reshape(img.data.cpu().numpy()[i], (200, 200)), cmap='gray')                a[1][i].clear()                a[1][i].imshow(np.reshape(label.data.cpu().numpy()[i], (200, 200)), cmap='gray')                a[2][i].clear()                a[2][i].imshow(np.reshape(output.data.cpu().numpy()[i], (200, 200)), cmap='gray')                pass            plt.draw()            plt.pause(0.05)            plt.savefig('./dc_img/image_{}.png'.format(epoch))    # ===================log========================    print('epoch [{}/{}], loss:{:.4f}, lr={}'          .format(epoch+1, num_epochs, total_loss, optimizer.param_groups[0]['lr']))torch.save(model.state_dict(), './conv_autoencoder.pth')