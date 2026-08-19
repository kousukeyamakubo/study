import serial
import cv2
import time

# 接続するT14REのキャプチャIDの設定
CaptureDeviceIndex = 1
# 接続するT14REのMMICコマンドインターフェースの設定
MmicCommandPort = 'COM19'
# コンフィグファイルの設定
CfgFile = 'T14RE_3D_Short.cfg'

# 取り込むデータサイズの指定
Sample = 256
Tx = 3
Rx = 4
Chirpset = 16
Frame = 1

# 繰り返し回数の指定
Loop = 1



# MMICにコマンドを送信する
def SendCommand(serialCommand, command):
    # 改行文字をLFに変更して送信
    command = command.replace('\r', '')
    command = command.replace('\n', '')
    command += '\n'
    serialCommand.write(command.encode())
    
    # 応答待ち
    ret = serialCommand.read_until(b'\nmmwDemo:/>')
    print(ret)

    # 応答確認
    # 空白、コメント、sensorResetはDoneを返さない
    if (
        (command.strip() != '') and
        (command.strip()[0] != '%') and
        (command != 'sensorReset\n')
    ):
        rets = ret.decode().split('\n')
        if (rets[len(rets) - 2] != 'Done'):
            return False

    return True



# コンフィグファイルをMMICに送信する
def SendCfg(filename):
    # Cfgファイルの内容を取得する
    file = open(filename)
    commands = file.readlines()
    file.close()

    # MMICコマンド用のシリアルポートを開く
    serialCommand = serial.Serial(MmicCommandPort, 115200)
    serialCommand.reset_output_buffer()
    serialCommand.reset_input_buffer()

    #コマンドの送信
    for command in commands:
        if SendCommand(serialCommand, command) == False:
            print("Error")
            serialCommand.close()
            return False
    
    serialCommand.close()
    return True





# CfgファイルをMMICに送信する
if (SendCfg(CfgFile) == False):
    exit()

# VideoCaptureオブジェクトを取得する
# Window: Microsoft Media Foundationを使用する設定
capture = cv2.VideoCapture(CaptureDeviceIndex, cv2.CAP_MSMF)
capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
capture.set(cv2.CAP_PROP_FORMAT, -1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, Sample * 2 * Rx)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Tx * Chirpset * Frame)

for loop in range (Loop):
    # 取り込み
    ret, data = capture.read()

    # デコードしてCSV表示
    # MSMFでは1次元データフォーマットで取得できる
    if (ret):
        index = 0
        for fr in range(Frame):
            for cs in range(Chirpset):
                for tt in range(Tx):
                    for rr in range(Rx):
                        csv = str(fr) + "," + str(cs) + "," + str(tt) + "," + str(rr)
                        for ss in range(Sample):
                            valueI = data[0][index + 0] + data[0][index + 1] * 256
                            if (valueI >= 0x8000):
                                valueI -= 0x10000
                            
                            valueQ = data[0][index + 2] + data[0][index + 3] * 256
                            if (valueQ >= 0x8000):
                                valueQ -= 0x10000
                            
                            index += 4

                            csv += "," + str(valueI) + "," + str(valueQ)
                        print (csv)


# VideoCaptureを閉じる
capture.release()
