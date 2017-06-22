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
    neuralnet.cpp \
    comm_funcs.cpp

HEADERS += \
    neuron.h \
    layer.h \
    connection.h \
    neuralnet.h \
    comm_funcs.h
