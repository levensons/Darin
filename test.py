import numpy as np
import sys
import torch.nn as nn
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 225)

        return x


def Transform(Board, player):
    CurBoard = np.zeros(shape=(3, 15, 15), dtype=np.float)
    BlackBoard = np.zeros(shape=(15, 15), dtype=np.float)
    WhiteBoard = np.zeros(shape=(15, 15), dtype=np.float)

    if player == 1:
        FirstBoard = np.ones(shape=(15, 15), dtype=np.float)
    else:
        FirstBoard = -np.ones(shape=(15, 15), dtype=np.float)

    CurBoard[0, :] = FirstBoard

    for i in range(15):
        for j in range(15):
            if Board[i, j] == 1:
                BlackBoard[i, j] = 1
            if Board[i, j] == -1:
                WhiteBoard[i, j] = -1

    CurBoard[1, :] = BlackBoard
    CurBoard[2, :] = WhiteBoard
    return CurBoard


def check(Board, turn):
    Board *= turn
    BoardUp = np.ndarray(shape=(15, 15), dtype=np.int8)
    BoardLeft = np.ndarray(shape=(15, 15), dtype=np.int8)
    BoardDiag1 = np.ndarray(shape=(15, 15), dtype=np.int8)
    BoardDiag2 = np.ndarray(shape=(15, 15), dtype=np.int8)

    for i in range(Board.shape[0]):
        for j in range(Board.shape[1]):
            if i > 0:
                BoardUp[i, j] = BoardUp[i - 1, j] + Board[i, j]
                if i - 4 >= 0 and BoardUp[i, j] - BoardUp[i - 4, j] + Board[i - 4, j] == 4:
                    for k in range(5):
                        if Board[i - k, j] == 0:
                            Board *= turn
                            return i - k + 1, j + 1
            if j > 0:
                BoardLeft[i, j] = BoardLeft[i, j - 1] + Board[i, j]
                if j - 4 >= 0 and BoardLeft[i, j] - BoardLeft[i, j - 4] + Board[i, j - 4] == 4:
                    for k in range(5):
                        if Board[i, j - k] == 0:
                            Board *= turn
                            return i + 1, j - k + 1
            if i > 0 and j > 0:
                BoardDiag1[i, j] = BoardDiag1[i - 1, j - 1] + Board[i, j]
                if i - 4 >= 0 and j - 4 >= 0 and BoardDiag1[i, j] - BoardDiag1[i - 4, j - 4] + Board[i - 4, j - 4] == 4:
                    for k in range(5):
                        if Board[i - k, j - k] == 0:
                            Board *= turn
                            return i - k + 1, j - k + 1
            if i > 0 and j < 14:
                BoardDiag2[i, j] = BoardDiag2[i - 1, j + 1] + Board[i, j]
                if i - 4 >= 0 and j + 4 < 15 and BoardDiag2[i, j] - BoardDiag2[i - 4, j + 4] + Board[i - 4, j + 4] == 4:
                    for k in range(5):
                        if Board[i - k, j + k] == 0:
                            Board *= turn
                            return i - k + 1, j + k + 1

    Board *= turn
    return -1, -1


def to_move(pos):
    return idx2chr[pos[0]] + str(pos[1] + 1)


def to_pos(move):
    return chr2idx[move[0]], int(move[1:]) - 1


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx2chr = 'abcdefghjklmnop'
    chr2idx = {letter: pos for pos, letter in enumerate(idx2chr)}
    net = ConvNet()
    net = net.to(device)
    net.load_state_dict(torch.load("model5.dms", map_location=device))

    while True:
        gameStr = sys.stdin.readline()  # input()#
        if not gameStr:
            break
        gameStr = gameStr.strip().split()
        curBoard = np.zeros(shape=(15, 15), dtype=np.float)

        for i in range(len(gameStr)):
            x, y = to_pos(gameStr[i])
            if i % 2 == 0:
                curBoard[x, y] = 1
            else:
                curBoard[x, y] = -1

        color = len(gameStr) % 2

        board = Transform(curBoard, color)
        with torch.no_grad():
            outputs = net(torch.unsqueeze(torch.from_numpy(board).float(), 0))
            _, netTurn = torch.max(outputs, 1)
            netTurn = int(netTurn)
            turnX, turnY = netTurn // 15, netTurn % 15
            while curBoard[turnX, turnY] != 0:
                outputs[netTurn] = 0
                _, netTurn = torch.max(outputs, 1)
                netTurn = int(netTurn)
                turnX, turnY = netTurn // 15, netTurn % 15

        Test = check(curBoard, color)
        if Test != (-1, -1):
            turnX, turnY = Test

        myTurn = to_move((turnX, turnY))
        sys.stdout.write(myTurn + '\n')
        sys.stdout.flush()
