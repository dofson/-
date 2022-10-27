#include "widget.h"

#include <QApplication>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widget w;
    w.setWindowTitle("前景分割-柯老板手下");
    w.setFixedSize(1500, 750);
    w.show();
    return a.exec();
}
