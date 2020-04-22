from math import sqrt


# maxiumum likliehood function requires us the MLE values of mu (mean) and sigma (standard deviation)

def mean(nums):
    avg = 0;
    for i in nums:
        avg += i
    return avg / len(nums)

def std_dev(nums): #sqrt(sumation (x - u)^2 / N)
    avg = mean(nums)
    var = sum(pow(x-avg, 2) for x in nums)/len(nums)
    std = sqrt(var)

    return std;

## Maximum Likelihood values (aka log likelihood or liklihood distribution)
def mle(nums):
    return list([mean(nums), std_dev(nums)])


# calculate P(A|B) given P(A), P(B|A), P(B|not A)
# P(A|B = [P(B|A) * P(A)] / P(B)
def bayesian_inference(p_a, p_b_given_a, p_b_given_not_a):
    # probability not a
    p_not_a = 1 - p_a
    #P(B) probability of B
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    #posterior distribution
    posterior = (p_b_given_a * p_a) / p_b

    return posterior



if __name__ == "__main__":
    #Question 1
    probably_rain_given_clouds = bayesian_inference(.3, .95, .25) #probability of rain, probabilty of rain & clouds, probability of clouds no rain
    print(f'Question 1 {probably_rain_given_clouds}\n')

    # Question 2
    probability_mistakes_given_flagged = bayesian_inference(.15, .80, .05)
    print(f'Question 2 {probability_mistakes_given_flagged  }\n')

    # Question 3
    probably_bBlood_and_father = bayesian_inference(.75, .5, .09)
    print(f'Question 3 {probably_bBlood_and_father  }\n')

    # basically the probability of A given B is equal to 
    # 
    # the Probabiltiy of B given A, times the probability of A,
    # all divided by 
    # the Probability of B given A times Prbability of A, plus the probability of B given Not A times the probability of not A

    # Question 5
    probably