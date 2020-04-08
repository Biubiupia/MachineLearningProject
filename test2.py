from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QT_VERSION_STR
from PyQt5.Qt import PYQT_VERSION_STR
from sip import SIP_VERSION_STR
if __name__=='__main__':
    import sys
    app=QApplication(sys.argv)
    print("Qt5 Version Number is: {0}".format(QT_VERSION_STR))
    print("PyQt5 Version is: {}".format(PYQT_VERSION_STR))
    print("Sip Version is: {}".format(SIP_VERSION_STR))
    sys.exit(app.exec_())

# Qt5 Version Number is: 5.13.0
# PyQt5 Version is: 5.13.0
# Sip Version is: 4.19.18