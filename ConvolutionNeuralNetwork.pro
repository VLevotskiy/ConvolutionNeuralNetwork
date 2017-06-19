QT += core
QT -= gui

CONFIG += c++11

TARGET = ConvolutionNeuralNetwork
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    neuron__.cpp

HEADERS += \
    convolutionnn.h \
    neuron.h \
    random_num.h
