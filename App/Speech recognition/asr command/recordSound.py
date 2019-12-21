import os
import argparse
from readMicrophone import TapTester
import readMicrophone as mic
from KeyCodeModel import KeyCodeModel as k


if __name__=='__main__':
    tt = TapTester()
    wav = []
    count=0
    while True:
        count+=1
        tt.record(k.VK_UP,count)