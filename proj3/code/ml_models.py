import numpy as np
import torch
import torch.nn as nn
import time

class RotorEffectivenessPredictor(nn.Module):
    """
    Predicts effectiveness per rotor
    """
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        last_hidden = lstm_out.mean(dim=1)
        out = self.fc(last_hidden)
        return out


class RotorEffectivenessPredictor2(nn.Module):
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )

    def forward(self, x, hc=None):
        x = x.permute(0, 2, 1)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)

        if hc is None:
            lstm_out, _ = self.lstm(cnn_out)
        else:
            lstm_out, hc = self.lstm(cnn_out, hc)

        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


class BinaryFaultClassifier(nn.Module):
    def __init__(self, n_channels=6, hidden_size=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # self.lstm = nn.LSTM(
        #     input_size=128,
        #     hidden_size=hidden_size,
        #     num_layers=lstm_layers,
        #     batch_first=True,
        #     dropout=dropout if lstm_layers > 1 else 0,
        # )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            # nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = x.transpose(1, 2)
        cnn_out = self.conv(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        # weights = self.attn(lstm_out)
        # context = (lstm_out * weights).sum(dim=1)
        # out = self.fc(context)
        # last_hidden = lstm_out[:, -1, :]
        last_hidden = lstm_out.mean(dim=1)
        out = self.fc(last_hidden)
        return out.squeeze(1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regression model
# model = RotorEffectivenessPredictor().to(DEVICE)
# model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
# model.eval()

# Regression model - Unidirectional
model = RotorEffectivenessPredictor2().to(DEVICE)
model.load_state_dict(torch.load("best_model2.pt", map_location=DEVICE))
model.eval()

# Binary model
# model = BinaryFaultClassifier().to(DEVICE)
# model.load_state_dict(torch.load("best_binary_model.pt", map_location=DEVICE))
# model.eval()
# -----------------------------
WINDOW_LEN = 100
imu_buffer = []      # will store last 100 IMU readings
predictions = []     # list of (time, predicted effectiveness vector)
inference_times = [] # list of inference timing results

print("Model loaded and ready for inference!")

def process_imu_and_predict(curr_time, imu_sample):
    imu_buffer.append(imu_sample)

    if len(imu_buffer) > WINDOW_LEN:
        imu_buffer.pop(0)

    # Run prediction only when window is full
    if len(imu_buffer) == WINDOW_LEN:
        imu_window = np.array(imu_buffer, dtype=np.float32)

        # (1, W, 6) tensor
        x = torch.tensor(imu_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(x).cpu().numpy()[0]
        t_inf = (time.perf_counter() - start) * 1000  # ms

        predictions.append((curr_time, pred))
        inference_times.append(t_inf)

        # print(f"t={curr_time:.2f}s  eff={pred}  inference={t_inf:.3f}ms")

        return pred, t_inf

    return None, None

np.random.seed(0)