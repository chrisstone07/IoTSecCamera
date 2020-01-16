import RPi.GPIO as GPIO
import time
import os
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

## note=after calling GPIO.cleanup() function all data on display will get erase.
## You cannot use __send_data(), __send() functions and any variables used
## in all functions of this library. The functions which available for user is
## Print(note that capital P),clear,shift,begin,setCursor,blinkCursorOn,
## blinkCursorOff and The variables which available for user is right and left
## which used to indicated direction for shift the display.thanks for using
## this library. 

class lcd:
    right=True
    left=False
    __cmd=False
    __data=True
    def __send_data(self,value,signal):
        GPIO.output(self.__RS,signal)
        self.__send(value>>4)
        self.__send(value)
        time.sleep(0.001)        
        
    def __send(self,val):
        self.__val=val
        for i in range (0,4):
            GPIO.output(self.__D[i],((self.__val>>i)  & 0x01))
        GPIO.output(self.__EN,False)
        time.sleep(0.000001)
        GPIO.output(self.__EN,True)
        time.sleep(0.000001)
        GPIO.output(self.__EN,False)
        time.sleep(0.0001)

    def Print(self,text):
        self.__text=str(text)
	self.__length=len(self.__text)
        for i in range (0,self.__length):
            self.__a=ord(self.__text[i])
            self.__send_data(self.__a,self.__data)

    def clear(self):
        self.__send_data(0x01,self.__cmd)
		
    def setCursor(self,row,col):
        self.__col=col-1
        self.__row=row
        if(self.__row==1):
            self.__pos=0x80
        if(self.__row==2):
            self.__pos=0xC0
        self.__cursor=self.__pos+self.__col
        self.__send_data(self.__cursor,self.__cmd)   
        
    def begin(self,d4,d5,d6,d7,rs,en):
        self.__D=[d4,d5,d6,d7]
        self.__RS=rs
        self.__EN=en
        for i in range(0,4):
            GPIO.setup(self.__D[i],GPIO.OUT)
        GPIO.setup(self.__RS,GPIO.OUT)
        GPIO.setup(self.__EN,GPIO.OUT)
        time.sleep(0.050)
        self.__send_data(0x30,self.__cmd)##first try
        time.sleep(0.05)
        self.__send_data(0x30,self.__cmd)##sencond try
        time.sleep(0.05)
        self.__send_data(0x30,self.__cmd)##third try
        time.sleep(0.0015)
        self.__send_data(0x20,self.__cmd)##final go
        self.__send_data(0x28,self.__cmd)##select 4 bit, mode 2 lins ,5x7 font
        self.__send_data(0x01,self.__cmd)##clear screen
        self.__send_data(0x06,self.__cmd)##display ON
        self.__send_data(0x80,self.__cmd)## bring cursor to position 0 of line 1
        self.__send_data(0x0C,self.__cmd)## turn display ON for cursor blinking

    def shift(self,direction,count):
        self.__direction=direction
        self.__count=count
        if(self.__direction==self.left):
            for i in range (0,self.__count):
                self.__send_data(0x18,self.__cmd)
                time.sleep(1)
        if(self.__direction==self.right):
             for i in range (0,self.__count):
                self.__send_data(0x1C,self.__cmd)
                time.sleep(1)
    def blinkCursorOn(self):
	    self.__send_data(0x0F,self.__cmd)
		
    def blinkCursorOff(self):
	    self.__send_data(0x0C,self.__cmd)
