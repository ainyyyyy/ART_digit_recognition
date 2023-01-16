import sys
from PyQt5 import Qt, QtGui, QtCore
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
 


class relearnThread(Qt.QObject):
    finished = Qt.pyqtSignal(object)

    def __init__(self, images, r_kr, v, logger):
        super(relearnThread,self).__init__()
        self.images = images
        self.r_kr = r_kr
        self.v = v
        self.logger = logger

    def run(self):
        digits =  torch.from_numpy(pd.read_csv("datasets/MNIST/mnist_train_norm_dig.csv").values)
        vectors = torch.from_numpy(pd.read_csv("datasets/MNIST/mnist_train_norm_vect.csv").values)
        digits_v = torch.from_numpy(pd.read_csv("datasets/MNIST/mnist_test_norm_dig.csv").values) 
        vectors_v = torch.from_numpy(pd.read_csv("datasets/MNIST/mnist_test_norm_vect.csv").values) 
        digits = digits.to(device)
        vectors = vectors.to(device)
        digits_v = digits_v.to(device)
        vectors_v = vectors_v.to(device)

        x_arr = {'digits': digits, 'vectors': vectors.reshape(59999, 784)}
        ev_arr = {'digits': digits_v, 'vectors': vectors_v.reshape(9999, 784)}

        self.nn_tr = ART2(x_arr, ev_arr, images = self.images, r_kr = self.r_kr, v = self.v)
        self.nn_tr.training()
        self.logger.append('Точность распознавания составляет {}%'.format(round(self.nn_tr.evaluation() * 100, 2)))

        self.nn_tr.saveWeights('curr_weights.wts')
        self.finished.emit(True)


class GUIMainWindow(Qt.QMainWindow):
    def __init__(self):
        super().__init__()       
        x_arr = torch.ones(784, device = device)
        self.nn_rec = ART2()        
        self.nn_rec.recognize(x_arr)
        self.initUI()        

    def initUI(self):             
        self.setWindowTitle("Распознавание цифр")
        self.setGeometry(300, 300, 748, 448)

        self.im = Qt.QImage(448, 448, Qt.QImage.Format_RGB32)
        self.im.fill(Qt.Qt.white)
        self.begin_point_x = self.begin_point_y = None

        self.relearn_button = Qt.QPushButton('Запустить обучение', self)
        self.relearn_button.move(498, 300)
        self.relearn_button.resize(200, 30)
        self.relearn_button.setCheckable(True)
        self.relearn_button.clicked.connect(self.showParamsChooseWindow)
        
        self.rec_button = Qt.QPushButton('Распознать', self)
        self.rec_button.move(498, 350)        
        self.rec_button.resize(200, 30)
        self.rec_button.clicked.connect(self.recognize)

        self.delete_button = Qt.QPushButton('Удалить', self)
        self.delete_button.move(498, 400)        
        self.delete_button.resize(200, 30)
        self.delete_button.clicked.connect(self.delete)       

        self.nameLabel = Qt.QLabel(self)
        self.nameLabel.setText('Результат:')
        self.nameLabel.move(538, 220)

        self.line = Qt.QLineEdit(self)  
        self.line.setFont(QtGui.QFont("Times", 30, QtGui.QFont.Bold))
        self.line.setReadOnly(True)
        self.line.move(618, 200)
        self.line.resize(40, 60)
        
        self.logger = Qt.QTextEdit(self)
        self.logger.setReadOnly(True)
        self.logger.move(498, 20)
        self.logger.resize(200, 150)
        
        self.window2 = ParamsChooseWindow(self.logger)

    def mouseMoveEvent(self, event):        
        if (self.begin_point_x is None) & (self.begin_point_y is None):
            self.begin_point_x = event.x()
            self.begin_point_y = event.y()
        
        painter = Qt.QPainter(self.im)
        painter.setPen(Qt.QPen(Qt.Qt.black, 25, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        painter.drawLine(self.begin_point_x, self.begin_point_y, event.x(), event.y())
        painter.end()
        self.update()
        self.begin_point_x = event.x()
        self.begin_point_y = event.y()
    
    def mouseReleaseEvent(self, e):
        self.begin_point_x = self.begin_point_y = None    

    def paintEvent(self, event):
        canvasPainter = Qt.QPainter(self)
        canvasPainter.drawImage(0, 0, self.im)

    def recognize(self):
        self.im.save('image.jpeg')
        self.im1 = Image.open('image.jpeg')        
        self.im1 = self.im1.resize((28, 28))
        x_arr = 255 - torch.tensor(self.im1.getdata(), device = device)[:, 0]        
        x_arr = x_arr / (torch.sum(torch.square(x_arr)) ** 0.5)
        self.line.setText(str(int(self.nn_rec.recognize(x_arr))))

    def delete(self):
        self.im.fill(Qt.Qt.white)
        self.update()    
        
    def showParamsChooseWindow(self):              
        if self.window2.is_learning == False:  
            self.window2.show()      
        else:
            self.showThreadError()
         
    def showThreadError(self):        
        msg = Qt.QMessageBox()
        msg.setIcon(Qt.QMessageBox.Critical)
        msg.setText("Обучение уже запущено!")
        msg.setInformativeText('Дождитесь окончания процесса')
        msg.setWindowTitle("Ошибка")
        msg.exec_()
         
 
class ParamsChooseWindow(Qt.QWidget):
    def __init__(self, logger):
        super(ParamsChooseWindow, self).__init__()  
        self.initUI() 
        self.logger = logger

    def initUI(self):             
        self.setWindowTitle("Распознавание цифр")
        self.setGeometry(300, 300, 300, 350)

        self.ok_button = Qt.QPushButton('ОК', self)
        self.ok_button.move(50, 250)
        self.ok_button.resize(200, 30)
        self.ok_button.clicked.connect(self.getParams)
        self.ok_button.setCheckable(True)

        self.cancel_button = Qt.QPushButton('Отмена', self)
        self.cancel_button.move(50, 300)        
        self.cancel_button.resize(200, 30)
        self.cancel_button.clicked.connect(self.close)

        self.epLine = Qt.QLineEdit(self)   
        self.epLine.move(50, 125)
        self.epLine.resize(200, 30)  
        self.rLine = Qt.QLineEdit(self)   
        self.rLine.move(80, 185)
        self.rLine.resize(50, 30)  
        self.vLine = Qt.QLineEdit(self)   
        self.vLine.move(180, 185)
        self.vLine.resize(50, 30)        
        
        self.nameLabel = Qt.QLabel(self)
        self.nameLabel.setText('Введите значения параметров\nобучения')
        self.nameLabel.move(50, 30)
        self.epLabel = Qt.QLabel(self)
        self.epLabel.setText('Количество изображений:')
        self.epLabel.move(50, 100)
        self.rLabel = Qt.QLabel(self)
        self.rLabel.setText('R =')
        self.rLabel.move(50, 190)
        self.vLabel = Qt.QLabel(self)
        self.vLabel.setText('V =')
        self.vLabel.move(150, 190)

        self.is_learning = False

    def getParams(self):                  
        if (self.chekingParams()) and (self.images > 0) and (self.r_kr >= 0) and  (self.r_kr <= 1) and (self.v >= 0) and  (self.v <= 1):
            self.is_learning = True
            self.runThread()
        else:
            self.showParamError()
            return

    def chekingParams(self):          
        try:
            self.images = int(self.epLine.text())
            self.r_kr = float(self.rLine.text())
            self.v = float(self.vLine.text())
            return True
        except ValueError or AttributeError:
            return False

    def runThread(self):
        self.close()
        self.logger.setText('Выполняется обучение со следующими параметрами: {} изображений,\nr = {}, v = {}'.format(self.images, self.r_kr, self.v))
        self.thread = Qt.QThread(parent=self)
        self.rel_thr = relearnThread(self.images, self.r_kr, self.v, self.logger)
        self.rel_thr.moveToThread(self.thread)
        self.thread.started.connect(self.rel_thr.run)
        self.thread.start()   
        self.thread.finished.connect(self.finishLearning)
        self.thread.quit()

    def finishLearning(self):
        self.logger.append('Обучение завершено!') 
        self.is_learning = False

    def showParamError(self):        
        msg = Qt.QMessageBox()
        msg.setIcon(Qt.QMessageBox.Critical)
        msg.setText("Введите корректные значения!")
        msg.setInformativeText('(Значения параметров r и v должны быть в пределах от 0 до 1)')
        msg.setWindowTitle("Ошибка")
        msg.exec_()




class ART2:
    def __init__(self, x_arr = None, ev_arr = None, images = 0, r_kr = 0.8, v = 0.001):
        self.v = v
        self.x_arr = x_arr        
        self.ev_arr = ev_arr
        self.r_kr = r_kr
        self.success = 0
        self.success_v = 0        
        self.success_ev = 0
        self.success_arr = []        
        self.success_v_arr = []
        self.time_arr = []        
        self.time_arr_v = []
        self.images = images
        self.epochs = int(images / 59999) + 1

        if self.images == 0:
            self.neur_digit, self.neur_weights = self.loadWeights('curr_weights.wts')
            self.neur_digit = self.neur_digit.to(device=device)
            self.neur_weights = self.neur_weights.to(device=device)
        else:            
            self.neur_weights = self.x_arr['vectors'][0].unsqueeze(0)
            self.neur_digit = self.x_arr['digits'][0]

           
    def training(self, with_GUI = True):
        if with_GUI:
            for im in range (1, self.images):
                i = im % 59999
                self.success = self.recognize(self.x_arr['vectors'][i].unsqueeze(0), self.x_arr['digits'][i], self.success)
        else:    
            for im in range (1, self.images):
                i = im % 59999
                self.success = self.recognize(self.x_arr['vectors'][i].unsqueeze(0), self.x_arr['digits'][i], self.success)
                self.success_arr.append(self.success / im)
                self.time_arr.append(im)
                if im % (6 * self.epochs) == 0:
                    self.success_v = self.recognize(torch.reshape(self.ev_arr['vectors'][int(im / (6 * self.epochs))], (1, 784)), digit = self.ev_arr['digits'][int(im / (6 * self.epochs))], success = self.success_v, evaluation = True)
                    self.success_v_arr.append(6 * self.epochs * self.success_v / im)
                    self.time_arr_v.append(im)  
            
            self.accuracy = {'accuracy': self.success_arr, 
                         'accuracy_v': self.success_v_arr, 
                         'images': self.time_arr, 
                         'images_v': self.time_arr_v}
            self.showChart()
            return self.accuracy

    def evaluation(self):
        for im in range (9999):
            self.success_ev = self.recognize(torch.reshape(self.ev_arr['vectors'][im], (1, 784)), self.ev_arr['digits'][im], self.success_ev, evaluation = True)  
        total_accuracy = self.success_ev / 9999
        return total_accuracy
    
    def recognize(self, vector, digit = None, success = None, evaluation = False):
        self.r_arr = torch.sum(vector * self.neur_weights, axis = 1)
        if digit is None:
            return self.neur_digit[torch.argmax(self.r_arr)]
        if evaluation == False:
            if torch.max(self.r_arr) < self.r_kr:            
                self.neur_weights = torch.cat((self.neur_weights, vector), 0)
                self.neur_digit = torch.cat((self.neur_digit, digit))
            elif self.neur_digit[torch.argmax(self.r_arr)] == digit:
                self.neur_weights[torch.argmax(self.r_arr)] = (1 - self.v) * self.neur_weights[torch.argmax(self.r_arr)] + self.v * vector[0]
        
        if self.neur_digit[torch.argmax(self.r_arr)] == digit:
            success += 1
        return success
        
 
    def saveWeights(self, path): 
        with open(path,'wb') as f:
            pickle.dump([self.neur_digit, self.neur_weights], f)
                    
    def loadWeights(self, path):
        with open(path,'rb') as f:
            params = pickle.load(f,encoding='bytes')
        return params

    def showChart(self):
        plt.figure(figsize=(16,10), dpi=80)
        plt.plot('images', 'accuracy', data=self.accuracy)        
        plt.plot('images_v', 'accuracy_v', data=self.accuracy)
        plt.show()
 
        
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Qt.QApplication(sys.argv)
window = GUIMainWindow()
window.show()
app.exec_()
