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

#define NUM_OF_VALUES 100000
int main( int argc, const char * argv[]) {
    
    unsigned int numOfValues = NUM_OF_VALUES;
    float input[NUM_OF_VALUES];
    float output[NUM_OF_VALUES];
    
    for(int i = 0; i < NUM_OF_VALUES; i++)
        input[i] = rand() / (float)RAND_MAX;
    
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
    
    //now our executable is ready. however, executable will need the memory allocations that
    //need to be handled by the host (this side of code) because it uses data from global memory.
    //dynamic allocation can only be made by host
    cl_mem d_input, d_output;
    size_t dataSize = NUM_OF_VALUES * sizeof(float);
    
    //memory allocated to pass memory adresses as argument to the kernel function we will call.
    //here note that data in input array is copied to d_input while creating the buffer.
    d_input = clCreateBuffer( clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, input, NULL);
    d_output = clCreateBuffer( clContext, CL_MEM_WRITE_ONLY, dataSize, NULL, NULL);
    
    if( !d_input || !d_output){
        
        printf("Error during clCreateBuffer\n");
        return EXIT_FAILURE;
    }
    
    clerr = 0;
    clerr = clSetKernelArg( clKernel, 0, sizeof(cl_mem), (void *)&d_input);
    clerr &= clSetKernelArg( clKernel, 1, sizeof(cl_mem), (void *)&d_output);
    clerr &= clSetKernelArg( clKernel, 2, sizeof(unsigned int), (void *)&numOfValues);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clSetKernelArg\n");
        return EXIT_FAILURE;
    }
    
    for( int curDevice = 0; curDevice < numOfDevices; curDevice++){
        
        size_t localWorkGroupSize;
        clerr = clGetKernelWorkGroupInfo( clKernel, clDevices[curDevice], CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof(size_t), &localWorkGroupSize, NULL);
        printf("info: local work group size for device %d is %zu\n", curDevice, localWorkGroupSize);
        if( clerr != CL_SUCCESS){
            
            printf("Error during clGetKernelWorkGroupInfo for device id %d\n", curDevice);
            return EXIT_FAILURE;
        }
        
        //the only constraint for the global_work_size is that it must be a multiple of the
        //local_work_size (for each dimension).
        size_t globalWorkItems = (numOfValues/localWorkGroupSize + 1)*localWorkGroupSize;
        clerr = clEnqueueNDRangeKernel( clCommandQueues[curDevice], clKernel, 1, NULL, &globalWorkItems,
                                       &localWorkGroupSize, 0, NULL, NULL);
        
        if( clerr != CL_SUCCESS){
            
            printf("Error during clEnqueueNDRangeKernel\n");
            return EXIT_FAILURE;
        }
    }
    
    //block until all works in all queues are finished
    for( int curDevice = 0; curDevice < numOfDevices; curDevice++)
        clFinish(clCommandQueues[curDevice]);
    
    //for( int curDevice = 0; curDevice < numOfDevices && clerr == CL_SUCCESS; curDevice++)
        clerr = clEnqueueReadBuffer( clCommandQueues[numOfDevices-1], d_output, CL_TRUE, 0,
                                    sizeof(float)*numOfValues, output, 0, NULL, NULL);
    
    if( clerr != CL_SUCCESS){
        
        printf("Error during clEnqueueReadBuffer\n");
        return EXIT_FAILURE;
    }
    
    // Validate our results
    int correct = 0;
    for(int i = 0; i < numOfValues; i++)
    {
        if(output[i] == input[i] * input[i])
            correct++;
        else
            printf("%d %f %f\n", i, output[i], input[i]);
    }
    
    // Print a brief summary detailing the results
    printf("Computed '%d/%d' correct values!\n", correct, numOfValues);
    
    // Shutdown and cleanup
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseProgram(clProgram);
    clReleaseKernel(clKernel);
    for( int curDevice = 0; curDevice < numOfDevices; curDevice++)
        clReleaseCommandQueue(clCommandQueues[curDevice]);
    clReleaseContext(clContext);
    
    return 0;
}
