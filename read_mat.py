import pymatreader

folder = "C:\\Users\\xochipilli\\Desktop\\DMT\\CleanEEG\\CleanEEG\\"

bad_channels = pymatreader.read_mat(folder+"channels_removed.mat")
rejected_epochs = pymatreader.read_mat(folder+"rejected_epochs.mat")