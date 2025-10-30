# ----------------------------------------------------------------------------------------------------------------------
# GAUSSIAN PROCESS LMPC - THREE-WAY COMPARISON
# Compares: Standard LMPC vs Neural Network vs Gaussian Process
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')

import matplotlib.pyplot as plt
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC, MPCParams
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map
from NeuralNetworkDynamics import NNDynamicsPredictor
from GaussianProcessDynamics import GPDynamicsPredictor
import numpy as np
import os

def main():
    print("\n" + "="*100)
    print(" "*20 + "THREE-WAY COMPARISON: Standard vs Neural Network vs Gaussian Process")
    print("="*100 + "\n")
    
    # Parameters
    N = 14
    n = 6
    d = 2
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS = [x0, x0]
    dt = 0.1
    map = Map(0.4)
    vt = 0.8

    print(f"Configuration:")
    print(f"  - Track Length: {map.TrackLength:.2f} m")
    print(f"  - Target Velocity: {vt} m/s\n")

    # Initialize
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)
    simulator = Simulator(map)
    LMPCsimulator = Simulator(map, multiLap=False, flagLMPC=True)

    os.makedirs("models", exist_ok=True)
    
    # Initialize BOTH ML models
    print("="*100)
    print("INITIALIZING ML MODELS")
    print("="*100 + "\n")
    
    nn_predictor = NNDynamicsPredictor(state_dim=n, input_dim=d, hidden_dim=128, learning_rate=1e-3, device='cpu')
    gp_predictor = GPDynamicsPredictor(state_dim=n, input_dim=d, noise_level=0.01)
    
    print("✓ Neural Network initialized")
    print("✓ Gaussian Process initialized\n")

    # Storage
    lap_times = {
        'PID': None,
        'MPC': None,
        'TV-MPC': None,
        'LMPC_Standard': [],
        'LMPC_NN': [],
        'LMPC_GP': []
    }

    # PHASE 1-3: Collect baseline data
    print("="*100)
    print("PHASE 1: PID CONTROLLER")
    print("="*100)
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    lap_times['PID'] = xPID_cl.shape[0] * dt
    print(f"✓ PID: {lap_times['PID']:.2f}s\n")
    
    nn_predictor.add_trajectory(xPID_cl, uPID_cl)
    gp_predictor.add_trajectory(xPID_cl, uPID_cl)

    print("="*100)
    print("PHASE 2: MPC")
    print("="*100)
    A, B, _ = Regression(xPID_cl, uPID_cl, 0.0000001)
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    lap_times['MPC'] = xMPC_cl.shape[0] * dt
    print(f"✓ MPC: {lap_times['MPC']:.2f}s\n")
    
    nn_predictor.add_trajectory(xMPC_cl, uMPC_cl)
    gp_predictor.add_trajectory(xMPC_cl, uMPC_cl)

    print("="*100)
    print("PHASE 3: TV-MPC")
    print("="*100)
    predictiveModel = PredictiveModel(n, d, map, 1)
    predictiveModel.addTrajectory(xPID_cl, uPID_cl)
    ltvmpcParam.timeVarying = True 
    mpc = MPC(ltvmpcParam, predictiveModel)
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc)
    lap_times['TV-MPC'] = xTVMPC_cl.shape[0] * dt
    print(f"✓ TV-MPC: {lap_times['TV-MPC']:.2f}s\n")
    
    nn_predictor.add_trajectory(xTVMPC_cl, uTVMPC_cl)
    gp_predictor.add_trajectory(xTVMPC_cl, uTVMPC_cl)

    # TRAIN BOTH ML MODELS
    print("="*100)
    print("TRAINING ML MODELS")
    print("="*100 + "\n")
    
    print("Training Neural Network...")
    nn_predictor.train(epochs=100, batch_size=64, verbose=True)
    nn_predictor.save_model("models/nn_initial.pth")
    
    print("\nTraining Gaussian Process...")
    gp_predictor.train(verbose=True)
    gp_predictor.save_model("models/gp_initial.pkl")

    # PHASE 4: STANDARD LMPC
    print("="*100)
    print("PHASE 4: STANDARD LMPC")
    print("="*100 + "\n")
    
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for i in range(4):
        lmpcpredictiveModel.addTrajectory(xPID_cl, uPID_cl)
    lmpcParameters.timeVarying = True 
    lmpc_standard = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for i in range(4):
        lmpc_standard.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    xS_std = [x0, x0]
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS_std = LMPCsimulator.sim(xS_std, lmpc_standard)
        lmpc_standard.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)
        lap_times['LMPC_Standard'].append(lmpc_standard.Qfun[it][0] * dt)
        print(f"  Lap {it}: {lap_times['LMPC_Standard'][-1]:.2f}s")
    
    print(f"\n✓ Standard LMPC best: {min(lap_times['LMPC_Standard']):.2f}s\n")

    # PHASE 5: NEURAL NETWORK LMPC
    print("="*100)
    print("PHASE 5: NEURAL NETWORK LMPC")
    print("="*100 + "\n")
    
    xS_nn = [x0, x0]
    LMPCsimulator_NN = Simulator(map, multiLap=False, flagLMPC=True)
    lmpcpredictiveModel_NN = PredictiveModel(n, d, map, 4)
    for i in range(4):
        lmpcpredictiveModel_NN.addTrajectory(xPID_cl, uPID_cl)
    lmpc_nn = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel_NN)
    for i in range(4):
        lmpc_nn.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    for it in range(numSS_it, Laps):
        xLMPC_NN, uLMPC_NN, xLMPC_NN_glob, xS_nn = LMPCsimulator_NN.sim(xS_nn, lmpc_nn)
        lmpc_nn.addTrajectory(xLMPC_NN, uLMPC_NN, xLMPC_NN_glob)
        lmpcpredictiveModel_NN.addTrajectory(xLMPC_NN, uLMPC_NN)
        nn_predictor.add_trajectory(xLMPC_NN, uLMPC_NN)
        
        if it % 3 == 0 and it > numSS_it:
            print(f"  → Retraining Neural Network...")
            nn_predictor.train(epochs=20, batch_size=64, verbose=False)
        
        lap_times['LMPC_NN'].append(lmpc_nn.Qfun[it][0] * dt)
        print(f"  Lap {it}: {lap_times['LMPC_NN'][-1]:.2f}s")
    
    print(f"\n✓ Neural Network LMPC best: {min(lap_times['LMPC_NN']):.2f}s\n")
    nn_predictor.save_model("models/nn_final.pth")

    # PHASE 6: GAUSSIAN PROCESS LMPC
    print("="*100)
    print("PHASE 6: GAUSSIAN PROCESS LMPC")
    print("="*100 + "\n")
    
    xS_gp = [x0, x0]
    LMPCsimulator_GP = Simulator(map, multiLap=False, flagLMPC=True)
    lmpcpredictiveModel_GP = PredictiveModel(n, d, map, 4)
    for i in range(4):
        lmpcpredictiveModel_GP.addTrajectory(xPID_cl, uPID_cl)
    lmpc_gp = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel_GP)
    for i in range(4):
        lmpc_gp.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    for it in range(numSS_it, Laps):
        xLMPC_GP, uLMPC_GP, xLMPC_GP_glob, xS_gp = LMPCsimulator_GP.sim(xS_gp, lmpc_gp)
        lmpc_gp.addTrajectory(xLMPC_GP, uLMPC_GP, xLMPC_GP_glob)
        lmpcpredictiveModel_GP.addTrajectory(xLMPC_GP, uLMPC_GP)
        gp_predictor.add_trajectory(xLMPC_GP, uLMPC_GP)
        
        if it % 3 == 0 and it > numSS_it:
            print(f"  → Retraining Gaussian Process...")
            gp_predictor.train(verbose=False)
        
        lap_times['LMPC_GP'].append(lmpc_gp.Qfun[it][0] * dt)
        print(f"  Lap {it}: {lap_times['LMPC_GP'][-1]:.2f}s")
    
    print(f"\n✓ Gaussian Process LMPC best: {min(lap_times['LMPC_GP']):.2f}s\n")
    gp_predictor.save_model("models/gp_final.pkl")

    # UNCERTAINTY ANALYSIS
    print("="*100)
    print("UNCERTAINTY ANALYSIS (GP Only)")
    print("="*100 + "\n")
    
    uncertainties = []
    for i in range(len(xLMPC_GP) - 1):
        _, std = gp_predictor.predict(xLMPC_GP[i], uLMPC_GP[i], return_std=True)
        avg_uncertainty = np.mean(std)
        uncertainties.append(avg_uncertainty)
    
    uncertainties = np.array(uncertainties)
    print(f"Average uncertainty: {np.mean(uncertainties):.6f}")
    print(f"Max uncertainty:     {np.max(uncertainties):.6f}")
    print(f"Min uncertainty:     {np.min(uncertainties):.6f}\n")

    # RESULTS COMPARISON
    print("="*100)
    print("FINAL THREE-WAY COMPARISON")
    print("="*100 + "\n")
    
    baseline = lap_times['PID']
    std_best = min(lap_times['LMPC_Standard'])
    nn_best = min(lap_times['LMPC_NN'])
    gp_best = min(lap_times['LMPC_GP'])
    
    std_avg = np.mean(lap_times['LMPC_Standard'])
    nn_avg = np.mean(lap_times['LMPC_NN'])
    gp_avg = np.mean(lap_times['LMPC_GP'])
    
    print("┌──────────────────────┬──────────────┬────────────────────┐")
    print("│       Method         │ Best Lap (s) │  Improvement       │")
    print("├──────────────────────┼──────────────┼────────────────────┤")
    print(f"│ PID (Baseline)       │    {baseline:6.2f}    │      0.0%          │")
    print(f"│ Standard LMPC        │    {std_best:6.2f}    │     {((baseline-std_best)/baseline)*100:5.1f}%          │")
    print(f"│ Neural Network LMPC  │    {nn_best:6.2f}    │     {((baseline-nn_best)/baseline)*100:5.1f}% 🧠       │")
    print(f"│ Gaussian Process LMPC│    {gp_best:6.2f}    │     {((baseline-gp_best)/baseline)*100:5.1f}% 🎯       │")
    print("└──────────────────────┴──────────────┴────────────────────┘\n")
    
    print("ML Method Comparison:")
    print(f"  NN vs Standard:  {((std_best - nn_best)/std_best)*100:+.2f}%")
    print(f"  GP vs Standard:  {((std_best - gp_best)/std_best)*100:+.2f}%")
    print(f"  GP vs NN:        {((nn_best - gp_best)/nn_best)*100:+.2f}%")
    
    if gp_best < nn_best:
        print(f"\n🎉 Gaussian Process WINS! ({((nn_best - gp_best)/nn_best)*100:.2f}% faster than NN)")
    elif nn_best < gp_best:
        print(f"\n🎉 Neural Network WINS! ({((gp_best - nn_best)/gp_best)*100:.2f}% faster than GP)")
    else:
        print(f"\n🤝 Tie! Both ML methods perform equally!")

    # PLOTTING
    print("\n" + "="*100)
    print("GENERATING PLOTS")
    print("="*100 + "\n")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    iterations = range(numSS_it, Laps)
    
    # Plot 1: Lap times
    ax1.plot(iterations, lap_times['LMPC_Standard'], 'o-', linewidth=2.5, markersize=8, 
             label='Standard', color='steelblue')
    ax1.plot(iterations, lap_times['LMPC_NN'], 's-', linewidth=2.5, markersize=8, 
             label='Neural Network', color='darkgreen')
    ax1.plot(iterations, lap_times['LMPC_GP'], '^-', linewidth=2.5, markersize=8, 
             label='Gaussian Process', color='darkorange')
    ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='PID', alpha=0.7)
    ax1.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Lap Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Three-Way Learning Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvements
    imp_std = [(lap_times['LMPC_Standard'][0] - t) for t in lap_times['LMPC_Standard']]
    imp_nn = [(lap_times['LMPC_NN'][0] - t) for t in lap_times['LMPC_NN']]
    imp_gp = [(lap_times['LMPC_GP'][0] - t) for t in lap_times['LMPC_GP']]
    
    ax2.plot(iterations, imp_std, 'o-', linewidth=2.5, markersize=8, color='steelblue')
    ax2.plot(iterations, imp_nn, 's-', linewidth=2.5, markersize=8, color='darkgreen')
    ax2.plot(iterations, imp_gp, '^-', linewidth=2.5, markersize=8, color='darkorange')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty
    ax3.plot(range(len(uncertainties)), uncertainties, '-', linewidth=2, color='darkorange')
    ax3.fill_between(range(len(uncertainties)), 0, uncertainties, alpha=0.3, color='darkorange')
    ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Uncertainty', fontsize=12, fontweight='bold')
    ax3.set_title('GP Uncertainty (Last Lap)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bar chart
    methods = ['Standard', 'Neural Net', 'Gaussian Proc']
    best_times = [std_best, nn_best, gp_best]
    colors = ['steelblue', 'darkgreen', 'darkorange']
    
    bars = ax4.bar(methods, best_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='PID', alpha=0.7)
    ax4.set_ylabel('Best Lap Time (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Best Lap Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, best_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Three_Way_Comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: Three_Way_Comparison.png\n")
    
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotClosedLoopLMPC(lmpc_standard, map)
    plotClosedLoopLMPC(lmpc_nn, map)
    plotClosedLoopLMPC(lmpc_gp, map)
    
    print("✓ All plots generated!")
    
    print("\n" + "="*100)
    print("THREE-WAY COMPARISON COMPLETE! 🎉")
    print("="*100 + "\n")
    
    plt.show()

if __name__== "__main__":
    main()
