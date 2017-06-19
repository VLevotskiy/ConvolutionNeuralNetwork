QT += core
QT -= gui

CONFIG += c++11

TARGET = ConvolutionNeuralNetwork
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    neuron.cpp \
    layer.cpp \
    connection.cpp \
    neuralnet.cpp

HEADERS += \
    convolutionnn.h \
    random_num.h \
    neuron.h \
    layer.h \
    connection.h \
    neuralnet.h
