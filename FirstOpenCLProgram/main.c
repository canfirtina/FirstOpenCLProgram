//
//  main.c
//  FirstOpenCLProgram
//
//  Created by Can Firtina on 22/07/15.
//  Copyright (c) 2015 Can Firtina. All rights reserved.
//

#include <stdio.h>
#include <OpenCL/OpenCL.h>

// Simple compute kernel which computes the square of an input array
const char *kernelSource = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char * argv[]) {
    
    cl_int clerr = CL_SUCCESS;
    
    //create context that includes gpu devices in it
    cl_context clContext = clCreateContextFromType( NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clCreateContextFromType\n");
        return EXIT_FAILURE;
    }
    
    //we will get the array of devices, but first we need to get the size (in bytes) of it
    size_t devicesSize;
    clerr = clGetContextInfo( clContext, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clGetContextInfo to get deviesSize\n");
        return EXIT_FAILURE;
    }
    
    //now we say that here is the array with devicesSize (in bytes). Fill it with devices
    cl_device_id *clDevices = (cl_device_id *)malloc(devicesSize);
    clerr = clGetContextInfo( clContext, CL_CONTEXT_DEVICES, devicesSize, clDevices, NULL);
    if( clerr != CL_SUCCESS) {
        
        printf("Error during clGetContextInfo to fill clDevices\n");
        return EXIT_FAILURE;
    }
    
    cl_uint numOfDevices;
    clerr = clGetContextInfo( clContext, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numOfDevices, NULL);
    
    //create the command queues of the devices. Note that each device needs its own command queue
    cl_command_queue *clCommandQueues = (cl_command_queue *)malloc(numOfDevices*sizeof(cl_command_queue));
    for( int curDevice = 0; curDevice < numOfDevices && clerr == CL_SUCCESS; curDevice++)
        clCommandQueues[curDevice] = clCreateCommandQueue( clContext, clDevices[curDevice], 0, &clerr);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clCommandQueues\n");
        return EXIT_FAILURE;
    }
    
    cl_program clProgram = clCreateProgramWithSource( clContext, 1, (const char **)&kernelSource, NULL, &clerr);
    
    if( clerr != CL_SUCCESS){
        //
        printf("Error during clCreateProgramWithSource\n");
        return EXIT_FAILURE;
    }
    
    //now the program object has been created with the source code specified for the devices
    //associated with the clProgram, which is also associated with clContext.
    clerr = clBuildProgram( clProgram, numOfDevices, clDevices, NULL, NULL, NULL);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clBuildProgram\n");
        return EXIT_FAILURE;
    }
    
    //Here program compiled. Now it needs to be executed. So we will create executable (kernel) for
    //that now... Worth to note that when you give a name that is different than the function name
    //that you have written for the kernel, clerr is not set CL_SUCCESS.
    cl_kernel clKernel = clCreateKernel( clProgram, "square", &clerr);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clCreateKernel\n");
        return EXIT_FAILURE;
    }
    
    
    return 0;
}
