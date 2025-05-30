#ifndef BAYES_FILTER_H
#define BAYES_FILTER_H

#include "rclcpp/rclcpp.hpp"
#include <thread>

class BayesFilter
{
    private:
        // Variables
        const double SIGMA_T = 0.1;
        const double LOCALIZATION_VARIANCE = 20;
        float cellElevation;
        float cellVariance;

    public:
        // Functions
        void updateCell(int, float);
        float getCellElevation();
        float getCellVariance();

        // Constructor
        BayesFilter();
};  

#endif