#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mpi.h"
#include <omp.h>
#define TASK_FACTOR 2 //indicate the portion of tasks that are not assign in the first place
#define TASK_TAG 101
#define FINISH_TAG 103 //indicate that there are no more task
#define ANSWER_TAG 102
#define DEBUG 0
//Global variables
int worldsize; //number of nodes
int myrank; //rank of this node
double real_lower;
double real_upper;
double img_lower;
double img_upper;
double real_step;
double img_step;
int num; //number of points to calculate
int maxiter; //maximum number of iteration for one number
int mandelbrotSetCount(double real_lower, double real_upper, double img_lower, double img_upper, int real_num, int img_num, int maxiter);
int init(){
    return 0;
}

//the master node will read the input and distribute to nodes
void master()
{
    //divide the task according to number of nodes
    //the master node is excluded 
    int totalCount = 0; //total inset number
    int taskCount = (worldsize-1) * TASK_FACTOR; //total number of tasks
    int pending = taskCount;//number of pending task (need to be distributed)
    int finished = 0; //number of finished task
    if(taskCount == 0){
        printf("There is only one node. Cannot implement master-slave. Exit.\n");
        exit(-1);
    }
    if(DEBUG){
        printf("I am the master node!\n");
        printf("There are %d nodes.\n",worldsize);
        printf("taskCount: %d\n",taskCount);
    }
    //task[0]: real_low [1]: real_up [2]: num (real)
    double current_position = real_lower;
    int taskSize = num / taskCount;
    double task[3];
    if(DEBUG)
        printf("Task size = %d\n",taskSize);
    for (int taskNum = 1; taskNum < taskCount; taskNum++){
        MPI_Request request;
        MPI_Status status;
        //make sure factor is not 1!
        task[0] = current_position;
        current_position += taskSize * real_step;
        task[1] = current_position;
        task[2] = (double)taskSize;
        if(DEBUG)
            printf("(%lf, %lf),num = %d\n", task[0],task[1],(int)task[2]);
        MPI_Isend(&task,3,MPI_DOUBLE,taskNum%(worldsize),TASK_TAG,MPI_COMM_WORLD,&request);
        MPI_Wait(&request,&status);
        pending--;
    }

    do{
        MPI_Request request;
        MPI_Status status;
        //receive answer from slave
        int tempCount = 0;
        if(finished == taskCount){
            //broadcast the finish signal to all nodes
            for(int rank = 1; rank < worldsize; rank++){
                double finish[3];
                MPI_Isend(&finish,3,MPI_DOUBLE,rank,FINISH_TAG,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,&status);
            } 
            printf("%d\n", totalCount);
            break;

        }
        MPI_Irecv(&tempCount,1,MPI_INT,MPI_ANY_SOURCE,ANSWER_TAG,MPI_COMM_WORLD,&request);
        MPI_Wait(&request,&status);
        int recvNode = status.MPI_SOURCE;
        if(DEBUG)
            printf("Receive answer from node %d\nCount = %d\n",recvNode,tempCount);
        totalCount+=tempCount;
        finished++;
        if(DEBUG){
            printf("pending = %d, finished = %d\n",pending,finished);
        }
        //if there are pending tasks, assign to the returned node
        if(pending > 0){
            task[0] = current_position;
            current_position += taskSize * real_step;
            //last task 
            if(current_position > real_upper){
                if(DEBUG)
                    printf("last task. \n");
                task[1] = real_upper;
                task[2] = (double)(num - taskSize * (taskCount-1));
                if(DEBUG)
                    printf("(%lf, %lf),num = %d\n", task[0],task[1],(int)task[2]);
                MPI_Isend(&task,3,MPI_DOUBLE,recvNode,TASK_TAG,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,&status);
                pending--;

            }
            //there is more than one task pending
            else{
                task[1] = current_position;
                task[2] = (double)taskSize; 
                if(DEBUG)
                    printf("(%lf, %lf),num = %d\n", task[0],task[1],(int)task[2]);
                MPI_Isend(&task,3,MPI_DOUBLE,recvNode,TASK_TAG,MPI_COMM_WORLD,&request);
                MPI_Wait(&request,&status);
                pending--;
            }
        }
    }while(finished <= taskCount);

}

//slave nodes
void slave(){
    if(DEBUG)
        printf("Hello from slave node: %d\n",myrank);
    MPI_Request request;
    MPI_Status status;
    double mytask[3];
    int myCount = 0;
    while(1){
        MPI_Irecv(&mytask,3,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&request);
        MPI_Wait(&request,&status);
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        if(DEBUG)
            printf("tag = %d\n",tag);
        if(tag == FINISH_TAG)
            break;
        
        else{

            if(DEBUG){
                printf("node %d Received task from node %d.\n",myrank,source);
                printf("Task detail:\n");
                printf("real_upper: %lf\n",mytask[0]);
                printf("real_lower: %lf\n",mytask[1]);
                printf("real_num: %d\n",(int)mytask[2]);
            }
            myCount = mandelbrotSetCount(mytask[0],mytask[1],img_lower,img_upper,(int)mytask[2],num,maxiter);
            if(DEBUG)
                printf("myCount: %d\n",myCount);
            MPI_Isend(&myCount,1,MPI_INT,0,ANSWER_TAG,MPI_COMM_WORLD,&request);
            MPI_Wait(&request,&status);
        }
    }
}

// return 1 if in set, 0 otherwise
int inset(double real, double img, int maxiter){
	double z_real = real;
	double z_img = img;
    #pragma omp parallel for
	for(int iters = 0; iters < maxiter; iters++){
		double z2_real = z_real*z_real-z_img*z_img;
		double z2_img = 2.0*z_real*z_img;
		z_real = z2_real + real;
		z_img = z2_img + img;
		if(z_real*z_real + z_img*z_img > 4.0) return 0;
	}
	return 1;
}

// count the number of points in the set, within the region
int mandelbrotSetCount(double real_lower, double real_upper, double img_lower, double img_upper, int real_num, int img_num, int maxiter){
	if(DEBUG)
        printf("%lf,%lf,%lf,%lf,%d,%d,%d",real_lower,real_upper,img_lower,img_upper,real_num,img_num,maxiter);
    int count=0;
    #pragma omp parallel for
	for(int real=0; real<real_num; real++){
		for(int img=0; img<img_num; img++){
			count+=inset(real_lower+real*real_step,img_lower+img*img_step,maxiter);
		}
	}
	return count;
}

// main
int main(int argc, char *argv[]){
	int num_regions = (argc-1)/6;
	for(int region=0;region<num_regions;region++){
		// scan the arguments
		sscanf(argv[region*6+1],"%lf",&real_lower);
		sscanf(argv[region*6+2],"%lf",&real_upper);
		sscanf(argv[region*6+3],"%lf",&img_lower);
		sscanf(argv[region*6+4],"%lf",&img_upper);
		sscanf(argv[region*6+5],"%i",&num);
		sscanf(argv[region*6+6],"%i",&maxiter);
    }
	//init();
	real_step = (real_upper-real_lower)/num;
	img_step = (img_upper-img_lower)/num;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(DEBUG){
        char hostname[256];
        gethostname(hostname,sizeof(hostname));
        printf("Hello from node: %d, hostname = %s, pid = %d\n",myrank,hostname,getpid());
    }
    if(myrank == 0) master();
    else slave();
    MPI_Finalize();
	return EXIT_SUCCESS;
}
