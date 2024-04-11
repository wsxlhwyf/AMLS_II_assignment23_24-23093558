import os

if __name__ == "__main__":
    #os.system('python ./A/A.py')
    for i in range(3):
        print("Task A and B have the same algorithm with different size of dataset")
        print("Task A: only part of the whole dataset(500) with shorter running time(40mins)")
        print("Task B: whole dataset(3605) with whole running time(18hrs) with GPU")
        print("Recommand to choose Task A --- does not require large memory")
        user_input = input("do u want to run A or B ?")
        if user_input == 'A':
            os.system('python ./A/A.py')
        elif user_input == 'B':
            os.system('python ./B/B.py')
        else:
            print('wrong mode!')