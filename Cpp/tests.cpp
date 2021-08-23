/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <vector>

using namespace std;

template<typename T>
void printVec(std::vector <T> const &a) {
   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
   std::cout << std::endl;
}

int main()
{
    // cout<<"Hello World";
    std::vector<int> outputs(10, 0);
    for (int i=0; i < outputs.size(); i++) outputs[i] = i;
    printVec(outputs);
    
    void* buffers[2];
    buffers[0] = malloc(784*sizeof(float));
    
    std::cout << buffers[1] << std::endl;
    std::cout << outputs.data() << std::endl;
    std::cout << &outputs.data()[1] << std::endl;
    
    buffers[1] = &outputs.data()[3];
    
    std::cout << ((int*)buffers[1])[0] << std::endl;
    
    free(buffers[0]);
    free(buffers[1]);
    

    return 0;
}
