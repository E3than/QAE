import numpy as np 
from qiskit.circuit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

#0.5 is chosen to represent the probability when flipping a coin 
p = .30

#setting up operators to be used in QAE --------------------------------------------------

class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(theta_p, 0)

class BernoulliQ(QuantumCircuit):
    """A circuit representing the Bernoulli Q operator."""

    def __init__(self, probability):
        super().__init__(1)  # circuit on 1 qubit

        self._theta_p = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self._theta_p, 0)

    def power(self, k):
        # implement the efficient power of Q
        q_k = QuantumCircuit(1)
        q_k.ry(2 * k * self._theta_p, 0)
        return q_k

A = BernoulliA(p)
Q = BernoulliQ(p)

#-----------------------------------------------------------------------------------------

#EstimationProblem object is instantiated, to be used in QAE algorithms 

from qiskit.algorithms import EstimationProblem

problem = EstimationProblem(
    state_preparation=A,  # A operator
    grover_operator=Q,  # Q operator
    objective_qubits=[0],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
)

#sampler is initialized

from qiskit.primitives import Sampler

sampler = Sampler()

#-----------------------------------------------------------------------------------------

#AmplitudeEstimation object created to be used in Canonical Quantum Amplitude Estimation 

from qiskit.algorithms import AmplitudeEstimation

ae = AmplitudeEstimation(
    num_eval_qubits=3,  # the number of evaluation qubits specifies circuit width and accuracy
    sampler=sampler,
)

ae_result = ae.estimate(problem)

print("Results from QAE Algorithms: ")

#collect information

ae_confidence_interval = ae.compute_confidence_interval(ae_result,0.15,"fisher")

ae_mle = ae.compute_mle(ae_result,False)

#display information

print("Canonical Result = ", ae_result.estimation)

print("Canonical Result Using MaximumLikelihoodEstimation Method = ", ae_mle)

print("Canonical Result Confidence Interval = ", ae_confidence_interval)

#construct and display circuit 

ae_circuit = ae.construct_circuit(problem,True)

ae_circuit.decompose().decompose().draw("mpl")

plt.show()

#-----------------------------------------------------------------------------------------

#IterativeAmplitudeEstimation object created to be used in Iterative Quantum Amplitude Estimation 

from qiskit.algorithms import IterativeAmplitudeEstimation

iae = IterativeAmplitudeEstimation(
    epsilon_target=0.01,  # target accuracy
    alpha=0.05,  # width of the confidence interval
    sampler=sampler,
)

#MaximumAmplitudeEstimation object created to be used in Maximum Quantum Amplitude Estimation 

from qiskit.algorithms import MaximumLikelihoodAmplitudeEstimation

mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=3,  # log2 of the maximal Grover power
    sampler=sampler,
)



#Implementation for Iterative Amplitude Estimation
iae_result = iae.estimate(problem)

#display information

print("Iterative Result = ",iae_result.estimation)

#Implementation for Maximum Likelihood Amplitude Estimation
mlae_result = mlae.estimate(problem)

mlae_confidence_interval = mlae.compute_confidence_interval(mlae_result,0.15,"fisher",False)

#display information

print("Maximum Likelihood Result = ",mlae_result.estimation)

print("Maximum Likelihood Result Confidence Interval = ", mlae_confidence_interval)


