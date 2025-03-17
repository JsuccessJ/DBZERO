import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

# 재귀 대신 반복문을 사용한 역전파 계산산
    def backward(self):
        funcs = [self.creator] # 함수 목록 생성
        while funcs: # 함수 목록이 빌 때까지 반복
            f = funcs.pop()  # 1. Get a function 
            x, y = f.input, f.output  # 2. Get the function's input/output
            x.grad = f.backward(y.grad)  # 3. Call the function's backward

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data # input은 Variable 객체
        y = self.forward(x)  # forward 메서드 호출
        output = Variable(y) # 새로운 Variable 객체 생성
        output.set_creator(self) # 생성자 설정
        self.input = input # 입력 변수 저장
        self.output = output # 출력 변수 저장
        return output # 새로운 Variable 객체 반환

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)  # A는 Square 클래스의 인스턴스
b = B(a)
y = C(b)

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)