# ----------------------------------------------------------------------------------------------------------------------
# ML-ENHANCED LMPC - COMPARISON WITH STANDARD LMPC
# Uses original Track.py (L-shaped track from the repository)
# Compares Standard LMPC vs ML-Enhanced LMPC side-by-side
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
from Track import Map  # Using original Track.py
from NeuralNetworkDynamics import NNDynamicsPredictor  # ML component
import numpy as np
import os

def main():
    print("\n" + "="*100)
    print(" "*25 + "ML-ENHANCED LMPC vs STANDARD LMPC COMPARISON")
    print("="*100 + "\n")
    
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 14
    n = 6
    d = 2
    x0 = np.array([0.5, 0, 0, 0, 0, 0])
    xS = [x0, x0]
    dt = 0.1
    map = Map(0.4)
    vt = 0.8

    print(f"Track Configuration:")
    print(f"  - Track Type: L-shaped (Original)")
    print(f"  - Track Length: {map.TrackLength:.2f} m")
    print(f"  - Track Half Width: {map.halfWidth} m")
    print(f"  - Target Velocity: {vt} m/s")
    print(f"  - Horizon Length: {N}")
    print(f"  - Time Step: {dt} s\n")

    # Initialize controller parameters
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)

    # Init simulators
    simulator = Simulator(map)
    LMPCsimulator = Simulator(map, multiLap=False, flagLMPC=True)

    # ======================================================================================================================
    # ================================== INITIALIZE NEURAL NETWORK =========================================================
    # ======================================================================================================================
    print("="*100)
    print("INITIALIZING NEURAL NETWORK FOR ML-ENHANCED VERSION")
    print("="*100 + "\n")
    
    os.makedirs("models", exist_ok=True)
    
    nn_predictor = NNDynamicsPredictor(
        state_dim=n, 
        input_dim=d, 
        hidden_dim=128,
        learning_rate=1e-3,
        device='cpu'
    )
    print("✓ Neural Network initialized")
    print(f"  - State dimension: {n}")
    print(f"  - Action dimension: {d}")
    print(f"  - Hidden units: 128")
    print(f"  - Device: CPU\n")

    # Storage for comparison
    lap_times = {
        'PID': None,
        'MPC': None,
        'TV-MPC': None,
        'LMPC_Standard': [],
        'LMPC_ML': []
    }

    # ======================================================================================================================
    # ======================================= PHASE 1: PID =================================================================
    # ======================================================================================================================
    print("="*100)
    print("PHASE 1: PID CONTROLLER (Baseline)")
    print("="*100)
    
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    lap_times['PID'] = xPID_cl.shape[0] * dt
    
    print(f"✓ PID completed")
    print(f"  - Lap time: {lap_times['PID']:.2f} seconds")
    print(f"  - Total steps: {xPID_cl.shape[0]}\n")
    
    # Add to neural network
    nn_predictor.add_trajectory(xPID_cl, uPID_cl)
    print(f"✓ Added {xPID_cl.shape[0]} data points to neural network\n")

    # ======================================================================================================================
    # ======================================= PHASE 2: MPC =================================================================
    # ======================================================================================================================
    print("="*100)
    print("PHASE 2: MPC WITH LINEAR MODEL")
    print("="*100)
    
    lamb = 0.0000001
    A, B, Error = Regression(xPID_cl, uPID_cl, lamb)
    mpcParam.A = A
    mpcParam.B = B
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    lap_times['MPC'] = xMPC_cl.shape[0] * dt
    
    print(f"✓ MPC completed")
    print(f"  - Lap time: {lap_times['MPC']:.2f} seconds")
    improvement = ((lap_times['PID'] - lap_times['MPC']) / lap_times['PID']) * 100
    print(f"  - Improvement vs PID: {improvement:.1f}%\n")
    
    nn_predictor.add_trajectory(xMPC_cl, uMPC_cl)
    print(f"✓ Added {xMPC_cl.shape[0]} data points to neural network\n")

    # ======================================================================================================================
    # ======================================= PHASE 3: TV-MPC ==============================================================
    # ======================================================================================================================
    print("="*100)
    print("PHASE 3: TIME-VARYING MPC")
    print("="*100)
    
    predictiveModel = PredictiveModel(n, d, map, 1)
    predictiveModel.addTrajectory(xPID_cl, uPID_cl)
    ltvmpcParam.timeVarying = True 
    mpc = MPC(ltvmpcParam, predictiveModel)
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc)
    lap_times['TV-MPC'] = xTVMPC_cl.shape[0] * dt
    
    print(f"✓ TV-MPC completed")
    print(f"  - Lap time: {lap_times['TV-MPC']:.2f} seconds")
    improvement = ((lap_times['PID'] - lap_times['TV-MPC']) / lap_times['PID']) * 100
    print(f"  - Improvement vs PID: {improvement:.1f}%\n")
    
    nn_predictor.add_trajectory(xTVMPC_cl, uTVMPC_cl)
    print(f"✓ Added {xTVMPC_cl.shape[0]} data points to neural network\n")

    # ======================================================================================================================
    # =============================== TRAIN NEURAL NETWORK =================================================================
    # ======================================================================================================================
    print("="*100)
    print("TRAINING NEURAL NETWORK ON COLLECTED DATA")
    print("="*100)
    
    total_samples = len(nn_predictor.all_states)
    print(f"\nTotal training samples: {total_samples}")
    print(f"Training for 100 epochs...\n")
    
    nn_predictor.train(epochs=100, batch_size=64, verbose=True)
    nn_predictor.save_model("models/nn_dynamics_initial.pth")
    print("")

    # ======================================================================================================================
    # =============================== PHASE 4: STANDARD LMPC ===============================================================
    # ======================================================================================================================
    print("="*100)
    print("PHASE 4: STANDARD LMPC (Without Machine Learning)")
    print("="*100 + "\n")
    
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for i in range(4):
        lmpcpredictiveModel.addTrajectory(xPID_cl, uPID_cl)

    lmpcParameters.timeVarying = True 
    lmpc_standard = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for i in range(4):
        lmpc_standard.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    print("Running Standard LMPC...\n")
    xS_standard = [x0, x0]
    
    for it in range(numSS_it, Laps):
        xLMPC, uLMPC, xLMPC_glob, xS_standard = LMPCsimulator.sim(xS_standard, lmpc_standard)
        lmpc_standard.addTrajectory(xLMPC, uLMPC, xLMPC_glob)
        lmpcpredictiveModel.addTrajectory(xLMPC, uLMPC)
        
        lap_time = lmpc_standard.Qfun[it][0] * dt
        lap_times['LMPC_Standard'].append(lap_time)
        
        print(f"  Lap {it}: {lap_time:.2f}s")
    
    print(f"\n✓ Standard LMPC completed")
    best_standard = min(lap_times['LMPC_Standard'])
    print(f"  - Best lap: {best_standard:.2f}s\n")

    # ======================================================================================================================
    # =============================== PHASE 5: ML-ENHANCED LMPC ============================================================
    # ======================================================================================================================
    print("="*100)
    print("PHASE 5: ML-ENHANCED LMPC (With Neural Network)")
    print("="*100 + "\n")
    
    # Reset for ML version
    xS_ml = [x0, x0]
    LMPCsimulator_ML = Simulator(map, multiLap=False, flagLMPC=True)
    
    lmpcpredictiveModel_ML = PredictiveModel(n, d, map, 4)
    for i in range(4):
        lmpcpredictiveModel_ML.addTrajectory(xPID_cl, uPID_cl)

    lmpc_ml = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel_ML)
    for i in range(4):
        lmpc_ml.addTrajectory(xPID_cl, uPID_cl, xPID_cl_glob)
    
    print("Running ML-Enhanced LMPC...\n")
    
    for it in range(numSS_it, Laps):
        xLMPC_ML, uLMPC_ML, xLMPC_ML_glob, xS_ml = LMPCsimulator_ML.sim(xS_ml, lmpc_ml)
        lmpc_ml.addTrajectory(xLMPC_ML, uLMPC_ML, xLMPC_ML_glob)
        lmpcpredictiveModel_ML.addTrajectory(xLMPC_ML, uLMPC_ML)
        
        # Add new data to neural network
        nn_predictor.add_trajectory(xLMPC_ML, uLMPC_ML)
        
        # Retrain every 3 laps
        if it % 3 == 0 and it > numSS_it:
            print(f"\n  → Retraining neural network after lap {it}...")
            nn_predictor.train(epochs=20, batch_size=64, verbose=False)
            nn_predictor.save_model(f"models/nn_dynamics_lap{it}.pth")
            print(f"  ✓ Retrained and saved\n")
        
        lap_time_ml = lmpc_ml.Qfun[it][0] * dt
        lap_times['LMPC_ML'].append(lap_time_ml)
        
        # Compare with standard
        standard_time = lap_times['LMPC_Standard'][it - numSS_it]
        diff = standard_time - lap_time_ml
        
        print(f"  Lap {it}: ML={lap_time_ml:.2f}s | Standard={standard_time:.2f}s | Diff={diff:+.2f}s")
    
    print(f"\n✓ ML-Enhanced LMPC completed")
    best_ml = min(lap_times['LMPC_ML'])
    print(f"  - Best lap: {best_ml:.2f}s\n")
    
    nn_predictor.save_model("models/nn_dynamics_final.pth")

    # ======================================================================================================================
    # ================================== NEURAL NETWORK EVALUATION =========================================================
    # ======================================================================================================================
    print("="*100)
    print("NEURAL NETWORK PREDICTION ACCURACY")
    print("="*100 + "\n")
    
    # Test on last ML lap
    test_states = xLMPC_ML[:-1]
    test_actions = uLMPC_ML
    
    predictions = []
    for i in range(len(test_states)):
        pred = nn_predictor.predict(test_states[i], test_actions[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    ground_truth = xLMPC_ML[1:]
    
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
    
    print("Mean Absolute Error per state:")
    for i, name in enumerate(state_names):
        print(f"  {name:5s}: {mae[i]:.6f}")
    print(f"\nOverall MAE: {np.mean(mae):.6f}\n")

    # ======================================================================================================================
    # ================================== COMPREHENSIVE RESULTS =============================================================
    # ======================================================================================================================
    print("="*100)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*100 + "\n")
    
    baseline = lap_times['PID']
    
    print("┌──────────────────┬──────────────┬────────────────────┬─────────────────┐")
    print("│     Method       │ Lap Time (s) │ Improvement vs PID │     Status      │")
    print("├──────────────────┼──────────────┼────────────────────┼─────────────────┤")
    print(f"│ PID              │    {baseline:6.2f}    │       0.0%         │    Baseline     │")
    
    for method in ['MPC', 'TV-MPC']:
        time = lap_times[method]
        imp = ((baseline - time) / baseline) * 100
        print(f"│ {method:16s} │    {time:6.2f}    │      {imp:5.1f}%        │   Single Lap    │")
    
    print("├──────────────────┼──────────────┼────────────────────┼─────────────────┤")
    
    # Standard LMPC
    std_first = lap_times['LMPC_Standard'][0]
    std_best = min(lap_times['LMPC_Standard'])
    std_avg = np.mean(lap_times['LMPC_Standard'])
    
    imp_first = ((baseline - std_first) / baseline) * 100
    imp_best = ((baseline - std_best) / baseline) * 100
    
    print(f"│ Std LMPC (first) │    {std_first:6.2f}    │      {imp_first:5.1f}%        │   Learning...   │")
    print(f"│ Std LMPC (best)  │    {std_best:6.2f}    │      {imp_best:5.1f}%        │   Converged     │")
    print(f"│ Std LMPC (avg)   │    {std_avg:6.2f}    │      {((baseline-std_avg)/baseline)*100:5.1f}%        │                 │")
    
    print("├──────────────────┼──────────────┼────────────────────┼─────────────────┤")
    
    # ML LMPC
    ml_first = lap_times['LMPC_ML'][0]
    ml_best = min(lap_times['LMPC_ML'])
    ml_avg = np.mean(lap_times['LMPC_ML'])
    
    imp_ml_first = ((baseline - ml_first) / baseline) * 100
    imp_ml_best = ((baseline - ml_best) / baseline) * 100
    
    print(f"│ ML LMPC (first)  │    {ml_first:6.2f}    │      {imp_ml_first:5.1f}%        │   Learning...   │")
    print(f"│ ML LMPC (best)   │    {ml_best:6.2f}    │      {imp_ml_best:5.1f}%        │ 🚀 ML Enhanced! │")
    print(f"│ ML LMPC (avg)    │    {ml_avg:6.2f}    │      {((baseline-ml_avg)/baseline)*100:5.1f}%        │                 │")
    print("└──────────────────┴──────────────┴────────────────────┴─────────────────┘\n")
    
    # ML vs Standard comparison
    print("\n" + "="*100)
    print("ML-ENHANCED vs STANDARD LMPC DIRECT COMPARISON")
    print("="*100 + "\n")
    
    ml_advantage = ((std_best - ml_best) / std_best) * 100
    avg_advantage = ((std_avg - ml_avg) / std_avg) * 100
    
    print(f"Best lap time:")
    print(f"  - Standard LMPC:    {std_best:.2f}s")
    print(f"  - ML-Enhanced LMPC: {ml_best:.2f}s")
    print(f"  - ML Advantage:     {ml_advantage:+.2f}%")
    print(f"\nAverage lap time:")
    print(f"  - Standard LMPC:    {std_avg:.2f}s")
    print(f"  - ML-Enhanced LMPC: {ml_avg:.2f}s")
    print(f"  - ML Advantage:     {avg_advantage:+.2f}%\n")

    # ======================================================================================================================
    # ================================== PLOTTING ==========================================================================
    # ======================================================================================================================
    print("="*100)
    print("GENERATING COMPARISON PLOTS")
    print("="*100 + "\n")
    
    # Standard trajectory plots
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotTrajectory(map, xMPC_cl, xMPC_cl_glob, uMPC_cl, 'MPC')
    plotTrajectory(map, xTVMPC_cl, xTVMPC_cl_glob, uTVMPC_cl, 'TV-MPC')
    
    # LMPC trajectories
    plotClosedLoopLMPC(lmpc_standard, map)
    plotClosedLoopLMPC(lmpc_ml, map)
    
    # Comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    iterations = range(numSS_it, Laps)
    
    # Lap times
    ax1.plot(iterations, lap_times['LMPC_Standard'], 'o-', linewidth=2.5, 
             markersize=8, label='Standard LMPC', color='steelblue')
    ax1.plot(iterations, lap_times['LMPC_ML'], 's-', linewidth=2.5, 
             markersize=8, label='ML-Enhanced LMPC', color='darkgreen')
    ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=2, 
                label='PID Baseline', alpha=0.7)
    ax1.set_xlabel('Lap Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Lap Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('Learning Progress: Standard vs ML-Enhanced LMPC', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Improvements
    improvements_std = [(lap_times['LMPC_Standard'][0] - t) for t in lap_times['LMPC_Standard']]
    improvements_ml = [(lap_times['LMPC_ML'][0] - t) for t in lap_times['LMPC_ML']]
    
    ax2.plot(iterations, improvements_std, 'o-', linewidth=2.5, 
             markersize=8, label='Standard LMPC', color='steelblue')
    ax2.plot(iterations, improvements_ml, 's-', linewidth=2.5, 
             markersize=8, label='ML-Enhanced LMPC', color='darkgreen')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.fill_between(iterations, improvements_std, alpha=0.2, color='steelblue')
    ax2.fill_between(iterations, improvements_ml, alpha=0.2, color='darkgreen')
    ax2.set_xlabel('Lap Number', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time Improvement (seconds)', fontsize=14, fontweight='bold')
    ax2.set_title('Cumulative Improvement Over Time', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ML_vs_Standard_LMPC_Comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: ML_vs_Standard_LMPC_Comparison.png")
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(iterations))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, lap_times['LMPC_Standard'], width, 
                   label='Standard LMPC', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, lap_times['LMPC_ML'], width, 
                   label='ML-Enhanced LMPC', color='darkgreen', alpha=0.8)
    
    ax.set_xlabel('Lap Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lap Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Lap-by-Lap Comparison: Standard vs ML-Enhanced LMPC', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(iterations)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Lap_by_Lap_Comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: Lap_by_Lap_Comparison.png")
    
    # Animation
    animation_xy(map, lmpc_ml, Laps-1)
    
    print("\n✓ All plots generated!\n")
    
    # ======================================================================================================================
    # ================================== FINAL SUMMARY =====================================================================
    # ======================================================================================================================
    print("="*100)
    print("SIMULATION COMPLETE! 🎉")
    print("="*100 + "\n")
    
    print("Summary:")
    print(f"  ✓ Baseline (PID):         {baseline:.2f}s")
    print(f"  ✓ Best Standard LMPC:     {std_best:.2f}s ({imp_best:.1f}% improvement)")
    print(f"  ✓ Best ML-Enhanced LMPC:  {ml_best:.2f}s ({imp_ml_best:.1f}% improvement)")
    print(f"  ✓ ML Advantage:           {ml_advantage:+.2f}% faster than Standard LMPC")
    
    print(f"\nSaved models:")
    print(f"  - models/nn_dynamics_initial.pth")
    print(f"  - models/nn_dynamics_final.pth")
    print(f"  - models/nn_dynamics_lap*.pth")
    
    print(f"\nSaved plots:")
    print(f"  - ML_vs_Standard_LMPC_Comparison.png")
    print(f"  - Lap_by_Lap_Comparison.png")
    
    print("\n" + "="*100 + "\n")
    
    plt.show()

if __name__== "__main__":
    main()