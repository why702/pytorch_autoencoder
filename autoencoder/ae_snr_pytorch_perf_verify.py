import osimport shutilimport sysimport matplotlib.pyplot as pltimport numpy as npimport progressbarimport torchfrom torch import nnfrom torch.autograd import Functionfrom torch.autograd import Variablefrom torch.utils.tensorboard import SummaryWritersys.path.append("../pytorch_autoencoder")from utils.util import run_perf_sum_score, apply_perf_threadfrom utils.data_augmentation_fp import PerfDatasetfrom utils.combine_genuines_fpdbindex import parse_indexclass autoencoder(nn.Module):    def __init__(self):        super(autoencoder, self).__init__()        self.encoder = nn.Sequential(  # input shape (1, 200, 200)            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),        )        self.decoder = nn.Sequential(            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),            nn.ELU(),            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),            nn.ELU(),        )    def forward(self, x):        x = self.encoder(x)        x = self.decoder(x)        return xclass Perf_match(Function):    @staticmethod    def forward(ctx, label, input):        out = torch.ones_like(input)        # ctx.save_for_backward(input)        numpy_label = label.detach().numpy()        numpy_input = input.detach().numpy()        perf_result = torch.tensor(apply_perf_thread(numpy_label, numpy_input, 8))        perf_result = perf_result.unsqueeze(1)        perf_result = perf_result.unsqueeze(2)        perf_result = perf_result.unsqueeze(3)        out = torch.mul(out, perf_result)        return out    @staticmethod    def backward(ctx, grad_output):        return grad_output, grad_outputdef criterion_perf(label, input):    perf_result = Perf_match.apply(label, input)    target = torch.ones_like(input) * 60000    perf_loss = nn.L1Loss()(perf_result, target)    return perf_lossdef output_ae_imgs(fpdbindex_path, fpdbindex_data, PerfDataset, model):    root_dir = os.path.dirname(fpdbindex_path)    out_dir = root_dir + "_AE"    shutil.rmtree(out_dir, ignore_errors=True)    for fpdb in fpdbindex_data:        ipp_path = os.path.join(root_dir, fpdb['path'])        raw = torch.tensor(PerfDataset.get_img(ipp_path))        raw = raw.unsqueeze(0)        raw = raw.unsqueeze(1)        model.eval()        with torch.no_grad():            raw_ae = model(raw.to(device=device)).cpu()        raw_ae = torch.squeeze(raw_ae).detach().numpy()        out_path = os.path.join(out_dir, fpdb['path'])        PerfDataset.save_img(raw_ae, out_path)        pass    sum_score, score_array = run_perf_sum_score(out_dir)    print("test score = {}".format(sum_score))if __name__ == '__main__':    num_epochs = 100000    batch_size = 128    # learning_rate = 1e-3    learning_rate = 1e-5    img_width = 200    img_height = 200    model_width = 224    model_height = 224    pad_width = int((model_width - img_width) / 2)    pad_height = int((model_height - img_height) / 2)    np_pad = ((pad_height, model_height - img_height - pad_height), (pad_width, model_width - img_width - pad_width))    # get perf pair and result	gen_file = "G:\\git\\HISI_CH1AJA\\dec_egistec_200x200_cardo_CH1AJA_100K_R120_t2560000_dry94_L0\\NP_IPPNormal\\genuines.txt"	index_file = "G:\\git\\HISI_CH1AJA\\NP\\Normal\\i.fpdbindex"    output_file = 'pair_info.csv'    use_cuda = torch.cuda.is_available()    device = torch.device('cuda:0' if use_cuda else 'cpu')    if not os.path.exists('./dc_img'):        os.mkdir('./dc_img')    # get dataset    dataset = PerfDataset(gen_file=gen_file, index_file=index_file, csv_file=output_file, img_width=img_width,                          img_height=img_height, pad_width=np_pad, RBS=True, PI=False)    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)    fpdbindex_data = parse_index(index_file)    writer = SummaryWriter()    model = autoencoder().to(device=device)    criterion = nn.MSELoss()    weighting_file = './conv_autoencoder_1900.pth'    if os.path.exists(weighting_file):        model.load_state_dict(torch.load(weighting_file))        model.eval()        model.to(device)    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,                                 weight_decay=1e-5)    # and a learning rate scheduler    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,                                                   step_size=1000,                                                   gamma=0.5)    N_TEST_IMG = 5    f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))    plt.ion()  # continuously plot    for epoch in range(num_epochs):        train_loss = 0        total_perf_loss = 0        total_step = int(dataset.size / batch_size)        with progressbar.ProgressBar(max_value=total_step) as bar:            for step, (enroll_raw, enroll_ipp, verify_raw, verify_ipp, score) in enumerate(dataloader):                img_e = Variable(enroll_raw).to(device=device)                img_v = Variable(verify_raw).to(device=device)                label_e = Variable(enroll_ipp).to(device=device)                label_v = Variable(verify_ipp).to(device=device)                # ===================forward=====================                output_e = model(img_e)                output_v = model(img_v)                output_e = (output_e - torch.min(output_e)) / (torch.max(output_e) - torch.min(output_e)) * 255                output_v = (output_v - torch.min(output_v)) / (torch.max(output_v) - torch.min(output_v)) * 255                output_t = torch.cat((output_e, output_v), 0)                label_t = torch.cat((label_e, label_v), 0)                loss_t = criterion(output_t, label_t)                train_loss += loss_t.data                perf_loss = criterion_perf(label_e.cpu(), output_v.cpu()).to(device=device)                total_perf_loss += perf_loss.data                # ===================backward====================                optimizer.zero_grad()                # loss_t.backward()                loss_t.backward(retain_graph=True)                perf_loss.backward()                optimizer.step()                # print([x.grad for x in model.parameters()])                if step == 0:                    # show training result                    if epoch % 10 == 0:                        # plotting decoded image                        for i in range(N_TEST_IMG):                            a0 = np.reshape(img_v.data.cpu().numpy()[i], (model_width, model_height))                            a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a1 = np.reshape(label_v.data.cpu().numpy()[i], (model_width, model_height))                            a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a2 = np.reshape(output_v.data.cpu().numpy()[i], (model_width, model_height))                            a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a[0][i].clear()                            a[0][i].imshow(a0, cmap='gray')                            a[1][i].clear()                            a[1][i].imshow(a1, cmap='gray')                            a[2][i].clear()                            a[2][i].imshow(a2, cmap='gray')                            pass                        plt.draw()                        # plt.pause(0.05)                        plt.savefig('./dc_img/image_{}.png'.format(epoch))                        pass                bar.update(step)            pass  # 1 train epoch end        writer.add_scalar('Loss/train', loss_t.data, epoch)        # test        if epoch % 50 == 0:            output_ae_imgs(index_file, fpdbindex_data, dataset, model)        # # test        # with torch.no_grad():        #     test_loss = 0        #     output_ = None        #     if epoch % 50 == 0:        #         for step_, (img_, label_) in enumerate(dataloader_test):        #             img_ = Variable(img_).to(device=device)        #             label_ = Variable(label_).to(device=device)        #        #             # ===================forward=====================        #             output_ = model(img_)        #             output_ = (output_ - torch.min(output_)) / (torch.max(output_) - torch.min(output_)) * 255        #             loss_ = criterion(output_, label_)        #             test_loss += loss_        #        #         # plotting decoded image        #         for i in range(N_TEST_IMG):        #             a0 = np.reshape(img.data.cpu().numpy()[i], (model_width, model_height))        #             a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a1 = np.reshape(label.data.cpu().numpy()[i], (model_width, model_height))        #             a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a2 = np.reshape(output.data.cpu().numpy()[i], (model_width, model_height))        #             a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a[0][i].clear()        #             a[0][i].imshow(a0, cmap='gray')        #             a[1][i].clear()        #             a[1][i].imshow(a1, cmap='gray')        #             a[2][i].clear()        #             a[2][i].imshow(a2, cmap='gray')        #             pass        #         plt.draw()        #         # plt.pause(0.05)        #         plt.savefig('./dc_img_test/image_{}.png'.format(epoch))        #        #         writer.add_scalar('Loss/test', test_loss.data, epoch)        #         pass        lr_scheduler.step()        # ===================log========================        print('epoch [{}/{}], loss:{:.4f}, perf_loss:{:.4f}, lr={}'              .format(epoch + 1, num_epochs, train_loss, total_perf_loss, optimizer.param_groups[0]['lr']))        if epoch % 100 == 0 and epoch > 0:            torch.save(model.state_dict(), './model/conv_autoencoder_{}.pth'.format(epoch))            print('save ./model/conv_autoencoder_{}.pth'.format(epoch))    writer.close()