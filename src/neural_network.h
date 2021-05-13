#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK





/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int PatternCount = 3;
static const int InputNodes = 650;
static const int HiddenNodes = 16;
static const int OutputNodes = 3;
static const float LearningRate = 0.3;
static const float Momentum = 0.9;
static const float InitialWeightMax = 0.5;
static const float Success = 0.0004;


class NeuralNetwork {
    public:

        NeuralNetwork();
        // ~NeuralNetwork();

        void initWeights();
        void forward(const float Input[]);
        void backward(const float Input[], const float Target[]);

        float* get_output();

        float* get_HiddenWeights();
        float* get_OutputWeights();

        float get_error();


    private:



        float *Hidden = new float[HiddenNodes];
        float *Output = new float[OutputNodes];
        // float (*HiddenWeights)[HiddenNodes] = new float[InputNodes+1][HiddenNodes];
        float *HiddenWeights = new float[(InputNodes+1) * HiddenNodes]();
        // float (*OutputWeights)[OutputNodes] = new float[HiddenNodes+1][OutputNodes];
        float *OutputWeights = new float[(HiddenNodes+1) * OutputNodes]();
        float *HiddenDelta = new float[HiddenNodes];
        float *OutputDelta = new float[OutputNodes];
        // float (*ChangeHiddenWeights)[HiddenNodes] = new float[InputNodes+1][HiddenNodes];
        float *ChangeHiddenWeights = new float[(InputNodes+1) * HiddenNodes]();
        // float (*ChangeOutputWeights)[OutputNodes] = new float[HiddenNodes+1][OutputNodes];
        float *ChangeOutputWeights = new float[(HiddenNodes+1) * OutputNodes]();


        // float Hidden[HiddenNodes];
        // float Output[OutputNodes];
        // float HiddenWeights[InputNodes+1][HiddenNodes];
        // float OutputWeights[HiddenNodes+1][OutputNodes];
        // float HiddenDelta[HiddenNodes];
        // float OutputDelta[OutputNodes];
        // float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
        // float ChangeOutputWeights[HiddenNodes+1][OutputNodes];


        int i, j, p, q, r;
        int ReportEvery1000;
        long  TrainingCycle;
        float Rando;
        float Error;
        float Accum;
};


#endif
