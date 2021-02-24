import torchimport torchvisionfrom torch import nnfrom torch.autograd import Variablefrom torch.utils.data import DataLoaderimport torch.utils.data as Datafrom torch.autograd import Functionfrom torch import autogradfrom torchvision import transformsimport torchvision.transforms.functional as TFfrom torchvision.utils import save_imagefrom torchvision.datasets import MNISTimport osimport utilimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport randomimport cv2from numpy.fft import rfft2, irfft2import mathfrom torch.utils.tensorboard import SummaryWriteruse_cuda = torch.cuda.is_available()device = torch.device('cuda:0' if use_cuda else 'cpu')# device = 'cpu'debug = Falseif not os.path.exists('./dc_img'):    os.mkdir('./dc_img')num_epochs = 100000batch_size = 128learning_rate = 1e-3img_width = 200img_height = 200model_width = 224model_height = 224pad_width = int((model_width - img_width) / 2)pad_height = int((model_height - img_height) / 2)np_pad = ((pad_height, model_height - img_height - pad_height), (pad_width, model_width - img_width - pad_width))train_folder = "D:\\git\\20201021_ET713_3PG_A52_5G_MSS_v216.101_normal_21DB"test_folder = "D:\\git\\20201205_ET713_3PG_A52_5G_Chamber_3DB"output_folder = train_folder + "_AE"writer = SummaryWriter()class FaceLandmarksDataset(Data.Dataset):    """Face Landmarks dataset."""    def __init__(self, root_dir, csv_file, img_width, img_height, pad_width, transform=None):        """        Args:            csv_file (string): Path to the csv file with annotations.            root_dir (string): Directory with all the images.            transform (callable, optional): Optional transform to be applied                on a sample.        """        self.width = img_width        self.height = img_height        self.pad_width = pad_width        util.read_bins_toCSV(root_dir, csv_file, img_width, img_height, True)        self.landmarks_frame = pd.read_csv(csv_file)        self.size = self.landmarks_frame.shape[0]        self.root_dir = root_dir    def __len__(self):        return len(self.landmarks_frame)    def transform(self, image, label):        # # Resize        # resize = transforms.Resize(size=(520, 520))        # image = resize(image)        # label = resize(label)        #        # # Random crop        # i, j, h, w = transforms.RandomCrop.get_params(        #     image, output_size=(512, 512))        # image = TF.crop(image, i, j, h, w)        # label = TF.crop(label, i, j, h, w)        image = TF.to_pil_image(image)        label = TF.to_pil_image(label)        # Random horizontal flipping        if random.random() > 0.5:            image = TF.hflip(image)            label = TF.hflip(label)        # Random vertical flipping        if random.random() > 0.5:            image = TF.vflip(image)            label = TF.vflip(label)        # Transform to tensor        image = TF.to_tensor(image)        label = TF.to_tensor(label)        # image = TF.normalize(image, mean=(0,), std=(1,))        # label = TF.normalize(label, mean=(0,), std=(1,))        return image, label    def __getitem__(self, idx, trans=True):        if torch.is_tensor(idx):            idx = idx.tolist()        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), True)        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), True)        ipp = util.read_8bit_bin(self.landmarks_frame.iloc[idx, 2], (self.width, self.height), True)        # diff = util.subtract(image, bk)        util.mss_interpolation(image, self.width, self.height)        # plt.imshow(image)        # plt.show()        diff = image        # # normalize        # diff = ((diff - np.mean(diff)) / np.std(diff)).astype('float32')        # ipp = ((ipp - np.mean(ipp)) / np.std(ipp)).astype('float32')        # to uint8        diff = ((diff - np.min(diff)) / (np.max(diff) - np.min(diff))).astype('float32')        ipp = ipp.astype('float32')        diff = np.pad(diff, self.pad_width, 'reflect')        ipp = np.pad(ipp, self.pad_width, 'reflect')        if trans:            diff, ipp = self.transform(diff, ipp)        return diff, ipp    def get_img_path(self, idx):        if torch.is_tensor(idx):            idx = idx.tolist()        image = util.read_bin(self.landmarks_frame.iloc[idx, 0], (self.width, self.height), True)        bk = util.read_bin(self.landmarks_frame.iloc[idx, 1], (self.width, self.height), True)        ipp_path = self.landmarks_frame.iloc[idx, 2]        # diff = util.subtract(image, bk)        util.mss_interpolation(image, self.width, self.height)        diff = image        # # normalize        # diff = ((diff - np.mean(diff)) / np.std(diff)).astype('float32')        # to uint8        diff = ((diff - np.mean(diff)) / (np.max(diff) - np.min(diff))).astype('float32')        diff = np.pad(diff, self.pad_width, 'reflect')        diff = TF.to_tensor(diff)        diff.unsqueeze_(0)        return diff, ipp_pathdataset = FaceLandmarksDataset(root_dir=train_folder, csv_file='list.csv', img_width=img_width, img_height=img_height,                               pad_width=np_pad)dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)dataset_test = FaceLandmarksDataset(root_dir=test_folder, csv_file='list_test.csv', img_width=img_width,                                    img_height=img_height,                                    pad_width=np_pad)dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)class autoencoder(nn.Module):    def __init__(self):        super(autoencoder, self).__init__()        # self.encoder = nn.Sequential(  # input shape (1, 200, 200)        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=64, out_channels=200, kernel_size=5, stride=1, padding=2),        #     nn.ELU(),        # )        #        # self.decoder = nn.Sequential(        #     nn.Conv2d(in_channels=200, out_channels=64, kernel_size=5, stride=1, padding=2),        #     nn.ELU(),        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        #     nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),        #     nn.ELU(),        # )        self.encoder = nn.Sequential(  # input shape (1, 200, 200)            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),            # nn.Conv2d(in_channels=64, out_channels=200, kernel_size=3, stride=1, padding=2),            # nn.ELU(),        )        self.decoder = nn.Sequential(            # nn.Conv2d(in_channels=200, out_channels=64, kernel_size=3, stride=1, padding=2),            # nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),            nn.ELU(),        )    def forward(self, x):        x = self.encoder(x)        x = self.decoder(x)        return x# # loss from FFT## def LPF_Butterworth(width, height, kRadius, kOrder):#     fltDst = np.empty([height, width])#     cx = width / 2#     cy = height / 2#     for row in range(height):#         for col in range(width):#             kDistance = math.sqrt((col - cx) ** 2 + (row - cy) ** 2)#             fltDst[row][col] = 1 / (1 + pow((kDistance / kRadius),#                                             (2 * kOrder)))#     return fltDst### def HPF_Butterworth(width, height, kRadius, kOrder):#     fltDst = np.empty([height, width])#     cx = width / 2#     cy = height / 2#     for row in range(height):#         for col in range(width):#             kDistance = math.sqrt((col - cx) ** 2 + (row - cy) ** 2)#             fltDst[row][col] = 1 - 1 / (1 + pow((kDistance / kRadius),#                                                 (2 * kOrder)))#     return fltDst# inch2mm = 25.4# m_nDPI = 508# szImage = (min(model_width, model_height) * inch2mm) / m_nDPI# fcUp = int(szImage * 3 / 2)  # newborns baby# fcLow = int(szImage * 10 / 2)  # grown-ups# fltNoiseLow = LPF_Butterworth(model_width, model_height, fcLow, 4).astype(np.complex)# fltNoiseHigh = HPF_Butterworth(model_width, model_height, fcUp, 4)# def fft_snr(img):#     f = np.fft.fftshift(np.fft.fft2(img))#     f_l = f * fltNoiseLow#     f_hl = f_l * fltNoiseHigh#     img_hl = np.real(np.fft.ifft2(np.fft.ifftshift(f_hl)))#     img_n = img - img_hl#     img_n = torch.tensor(img_n)#     result = torch.std(img_n, dim=(1, 2, 3))#     if debug:#         util.show_ndarray(img[0, 0], "img")#         util.show_ndarray(20 * np.log(np.abs(f[0, 0])), "f")#         f_l_ = 20 * np.log(np.abs(f_l[0, 0] + 1))#         util.show_ndarray(f_l_, "f_l_")#         img_l = np.real(np.fft.ifft2(np.fft.ifftshift(f_l)))#         util.show_ndarray(img_l[0, 0], "img_l")#         f_hl_ = 20 * np.log(np.abs(f_hl[0, 0] + 1))#         util.show_ndarray(f_hl_, "f_hl_")#         util.show_ndarray(img_hl[0, 0], "img_hl")#         util.show_ndarray(img_n[0, 0], "img_n")#     return result### class FFT_NOISE(Function):#     @staticmethod#     def forward(ctx, input):#         out = torch.ones_like(input)#         ctx.save_for_backward(input)#         numpy_input = input.detach().numpy()#         result = fft_snr(numpy_input)#         result = result.unsqueeze(1)#         result = result.unsqueeze(2)#         result = result.unsqueeze(3)#         out = torch.mul(out, result)#         return out##     @staticmethod#     def backward(ctx, grad_output):#         return grad_output### class DRY_SCORE(Function):#     @staticmethod#     def forward(ctx, input):#         with torch.no_grad():#             numpy_input = input.detach().numpy()#             offset_height = int(model_height / 4)#             offset_width = int(model_width / 4)#             target_height = int(model_height / 2)#             target_width = int(model_width / 2)#             crop = numpy_input[:, :, offset_height:offset_height + target_height,#                    offset_width:offset_width + target_width]#             maxCorners = 1000#             qualityLevel = 0.1#             minDistance = 0.1#             corner_list = []#             for i in range(crop.shape[0]):#                 corners = cv2.goodFeaturesToTrack(crop[i, 0, :, :], maxCorners=maxCorners, qualityLevel=qualityLevel,#                                                   minDistance=minDistance)#                 if corners is not None:#                     corner_list.append(corners.size)#                 else:#                     corner_list.append(0)##         return input.new(np.array(corner_list))#         # return torch.as_tensor(np.array(corner_list))##     @staticmethod#     def backward(ctx, grad_output):#         return grad_output### class MSE_SCORE(Function):#     @staticmethod#     def forward(ctx, input, label):#         numpy_input = input.detach().numpy()#         numpy_label = label.detach().numpy()#         diff = (numpy_label - numpy_input) ** 2#         # mean = np.mean(diff, axis=(1,2,3))#         # mse = np.sum(mean)#         mse = np.sum(diff)#         ctx.save_for_backward(input, label)##         return torch.tensor(mse)##     @staticmethod#     def backward(ctx, grad_output):#         grad_output = grad_output.detach().numpy()#         input, label = ctx.saved_tensors#         numpy_input = input.detach().numpy()#         numpy_label = label.detach().numpy()#         grad = -2 * np.sum(np.mean((numpy_label - numpy_input), axis=(1, 2, 3))) * np.ones(input.shape) * grad_output#         return torch.tensor(grad), Nonemodel = autoencoder().to(device=device)criterion = nn.MSELoss(reduce=True, size_average=False)# weighting_file = './model/conv_autoencoder_600.pth'# if os.path.exists(weighting_file):#     model.load_state_dict(torch.load(weighting_file))#     model.eval()#     model.to(device)# def criterion_fft(input):#     fft_snr = FFT_NOISE.apply(input)#     target = torch.ones_like(input) * 5#     fft_loss = nn.L1Loss()(fft_snr, target)#     return fft_loss### def criterion_dry(input):#     return DRY_SCORE.apply(input)### def criterion_mse(input, label):#     return MSE_SCORE.apply(input, label)optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,                             weight_decay=1e-5)# and a learning rate schedulerlr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,                                               step_size=1000,                                               gamma=0.5)N_TEST_IMG = 5f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))plt.ion()  # continuously plotfor epoch in range(num_epochs):    train_loss = 0    # total_fft_loss = 0    # total_dry_loss = 0    for step, (img, label) in enumerate(dataloader):        img = Variable(img).to(device=device)        label = Variable(label).to(device=device)        # ===================forward=====================        output = model(img)        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output)) * 255        loss = criterion(output, label)        train_loss += loss.data        # ===================backward====================        optimizer.zero_grad()        loss.backward()        optimizer.step()        # print([x.grad for x in model.parameters()])        if step == 0:            writer.add_scalar('Loss/train', loss.data, epoch)            # show training result            if epoch % 10 == 0:                # plotting decoded image                for i in range(N_TEST_IMG):                    a0 = np.reshape(img.data.cpu().numpy()[i], (model_width, model_height))                    a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                    a1 = np.reshape(label.data.cpu().numpy()[i], (model_width, model_height))                    a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                    a2 = np.reshape(output.data.cpu().numpy()[i], (model_width, model_height))                    a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                    a[0][i].clear()                    a[0][i].imshow(a0, cmap='gray')                    a[1][i].clear()                    a[1][i].imshow(a1, cmap='gray')                    a[2][i].clear()                    a[2][i].imshow(a2, cmap='gray')                    pass                plt.draw()                # plt.pause(0.05)                plt.savefig('./dc_img/image_{}.png'.format(epoch))                pass            # test            with torch.no_grad():                test_loss = 0                output_ = None                if epoch % 50 == 0:                    for step_, (img_, label_) in enumerate(dataloader_test):                        img_ = Variable(img_).to(device=device)                        label_ = Variable(label_).to(device=device)                        # ===================forward=====================                        output_ = model(img_)                        output_ = (output_ - torch.min(output_)) / (torch.max(output_) - torch.min(output_)) * 255                        loss_ = criterion(output_, label_)                        test_loss += loss_                    # plotting decoded image                    for i in range(N_TEST_IMG):                        a0 = np.reshape(img.data.cpu().numpy()[i], (model_width, model_height))                        a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                        a1 = np.reshape(label.data.cpu().numpy()[i], (model_width, model_height))                        a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                        a2 = np.reshape(output.data.cpu().numpy()[i], (model_width, model_height))                        a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                        a[0][i].clear()                        a[0][i].imshow(a0, cmap='gray')                        a[1][i].clear()                        a[1][i].imshow(a1, cmap='gray')                        a[2][i].clear()                        a[2][i].imshow(a2, cmap='gray')                        pass                    plt.draw()                    # plt.pause(0.05)                    plt.savefig('./dc_img/image_{}_test.png'.format(epoch))                    writer.add_scalar('Loss/test', test_loss.data, epoch)                    writer.add_image('test', output_[0], epoch)                    pass    lr_scheduler.step()    # ===================log========================    # print('epoch [{}/{}], loss:{:.4f}, lr={}, fft loss:{:.6f}, dry loss:{:.6f}'    #       .format(epoch + 1, num_epochs, train_loss, optimizer.param_groups[0]['lr'], total_fft_loss, total_dry_loss))    print('epoch [{}/{}], loss:{:.4f}, lr={}'          .format(epoch + 1, num_epochs, train_loss, optimizer.param_groups[0]['lr']))    if epoch % 100 == 0 and epoch > 0:        torch.save(model.state_dict(), './model/conv_autoencoder_{}.pth'.format(epoch))        print('save ./model/conv_autoencoder_{}.pth'.format(epoch))writer.close()