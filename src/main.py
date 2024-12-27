import sys

from model import LeNet
import torch
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget, QMainWindow, QLabel, QVBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.setStyleSheet("background-color: #404a43;")

        self.setFixedSize(450, 450)
        self.setWindowTitle("LeNet Showcase")

        self.previousPoint = None

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.canvas = QPixmap(QSize(400, 400))
        self.canvas.fill(QColor("black"))

        self.pen = QPen()
        self.pen.setColor(Qt.GlobalColor.white)
        self.pen.setWidth(30)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        self.label.setPixmap(self.canvas)
        #self.setCentralWidget(self.label)

        self.button1 = QPushButton("Guess")
        self.button2 = QPushButton("Clear")

        self.button1.clicked.connect(self.guess)
        self.button2.clicked.connect(self.clear)

        self.lenetGuess = QLabel("LeNet's Guess: None")
        self.lenetGuess.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.lenetGuess)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        self.model = LeNet()
        self.model.load_state_dict(torch.load("LeNet_Params4.pt", weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

    def guess(self):
        image = self.canvas.toImage()
        image = image.convertToFormat(QImage.Format.Format_RGB888)

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(height * width * 3)
        img_as_np = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        #print(image)
        #print(img_as_np.shape)

        transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.Resize((28, 28)),
        ])

        output_tensor = transform(img_as_np)
        #print(output_tensor.shape)
        #print(output_tensor)

        plt.imshow(output_tensor.permute(1, 2, 0).numpy(), cmap="grey")
        plt.savefig('image.png')

        output = self.model(output_tensor.view(1, 1, 28, 28))
        guess = output.argmax().item()

        self.lenetGuess.setText(f"LeNet's Guess: {guess}")

    def clear(self):
        self.canvas.fill(QColor("black"))
        self.label.setPixmap(self.canvas)

    def mouseMoveEvent(self, event):
        position = event.pos()
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # Smooth drawing
        painter.setPen(self.pen)

        if self.previousPoint:
            painter.drawLine(self.previousPoint.x()-25, self.previousPoint.y()+5, position.x()-25, position.y()+5)
        else:
            painter.drawPoint(position.x()-25, position.y()+5)

        painter.end()

        self.label.setPixmap(self.canvas)
        self.previousPoint = position

    def mouseReleaseEvent(self, event):
        self.previousPoint = None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
