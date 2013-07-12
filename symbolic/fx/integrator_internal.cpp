/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "integrator_internal.hpp"
#include <cassert>
#include "../stl_vector_tools.hpp"
#include "../matrix/matrix_tools.hpp"
#include "../mx/mx_tools.hpp"
#include "../sx/sx_tools.hpp"
#include "mx_function.hpp"
#include "sx_function.hpp"

INPUTSCHEME(IntegratorInput)
OUTPUTSCHEME(IntegratorOutput)

using namespace std;
namespace CasADi{

  IntegratorInternal::IntegratorInternal(const FX& dae, int nfwd, int nadj) : dae_(dae), nfwd_(nfwd), nadj_(nadj){
    // set default options
    setOption("name","unnamed_integrator"); // name of the function 
  
    // Additional options
    addOption("print_stats",              OT_BOOLEAN,     false, "Print out statistics after integration");
    addOption("t0",                       OT_REAL,        0.0, "Beginning of the time horizon"); 
    addOption("tf",                       OT_REAL,        1.0, "End of the time horizon");
    addOption("fwd_via_sct",              OT_BOOLEAN,     true, "Generate new functions for calculating forward directional derivatives");
    addOption("adj_via_sct",              OT_BOOLEAN,     true, "Generate new functions for calculating adjoint directional derivatives");
    addOption("augmented_options",        OT_DICTIONARY,  GenericType(), "Options to be passed down to the augmented integrator, if one is constructed.");
  
    // Negative number of parameters for consistancy checking
    nfp_ = -1;
  
    inputScheme_ = SCHEME_IntegratorInput;
    outputScheme_ = SCHEME_IntegratorOutput;

    f_ = dae_;
  }

  IntegratorInternal::~IntegratorInternal(){ 
  }

  void IntegratorInternal::evaluate(int nfdir, int nadir){
    casadi_assert_message(adj_via_sct_,"Not implemented."); // NOTE: Currently not supported by any derived class

  
    // What needs to be calculated
    bool need_nondiff = true;
    bool need_fwd = nfdir!=0;
    bool need_adj = nadir!=0;
    
    // No sensitivity analysis
    bool no_sens = !need_fwd && !need_adj;
  
    // Calculate without source code transformation
    if(no_sens || (need_fwd && !fwd_via_sct_)){
  
      // Number of sensitivities integrating forward
      int nsens = fwd_via_sct_ ? 0 : nfdir; // NOTE: Can be overly pessimistic e.g. if there are no seeds at all in some directions
    
      // Number of sensitivities integrate_backward 
      int nsensB = nrx_>0 ? nsens : 0; // NOTE: Can be overly pessimistic e.g. if there are no seeds at all in some directions
    
      // Number of sensitivities in the forward integration to be used in the backward integration
      int nsensB_store = nsensB; // NOTE: Can be overly pessimistic e.g. if some sensitivities do not depend on the forward sensitivities
    
      // Reset solver
      reset(nsens,nsensB,nsensB_store);

      // Integrate forward to the end of the time horizon
      integrate(tf_);

      // If backwards integration is needed
      if(nrx_>0){
      
        // Re-initialize backward problem
        resetB();

        // Integrate backwards to the beginning
        integrateB(t0_);
      }
    
      // Mark to avoid overwriting
      need_nondiff = false;
      if(!fwd_via_sct_) need_fwd = false;
    }
  
    // Quick return if done
    if(!need_fwd && !need_adj) return;
  
    // Correct nfdir if needed
    if(!need_fwd) nfdir = 0;
  
    // Get derivative function
    FX dfcn = derivative(nfdir, nadir);

    int integ_in = NEW_INTEGRATOR_NUM_IN;
    int integ_out = NEW_INTEGRATOR_NUM_OUT;

    // Pass function values
    int input_index = 0;
    for(int i=0; i<integ_in; ++i){
      dfcn.setInput(input(i),input_index++);
    }
  
    // Pass forward seeds
    for(int dir=0; dir<nfdir; ++dir){
      for(int i=0; i<integ_in; ++i){
        dfcn.setInput(fwdSeed(i,dir),input_index++);
      }
    }
    
    // Pass adjoint seeds
    for(int dir=0; dir<nadir; ++dir){
      for(int i=0; i<integ_out; ++i){
        dfcn.setInput(adjSeed(i,dir),input_index++);
      }
    }
  
    // Evaluate to get function values and adjoint sensitivities
    dfcn.evaluate();
  
    // Get nondifferentiated results
    int output_index = 0;
    for(int i=0; i<integ_out; ++i){
      dfcn.getOutput(output(i),output_index++);
    }
  
    // Get forward sensitivities 
    for(int dir=0; dir<nfdir; ++dir){
      for(int i=0; i<integ_out; ++i){
        dfcn.getOutput(fwdSens(i,dir),output_index++);
      }
    }
  
    // Get adjoint sensitivities 
    for(int dir=0; dir<nadir; ++dir){
      for(int i=0; i<integ_in; ++i){
        dfcn.getOutput(adjSens(i,dir),output_index++);
      }
    }
  
    // Print statistics
    if(getOption("print_stats")) printStats(std::cout);
  
    //if (!integrator.isNull()) stats_["augmented_stats"] =  integrator.getStats();
  }

  void IntegratorInternal::init(){

    // Initialize the dae function and check signature
    casadi_assert(!dae_.isNull());
    dae_.init(false);
    casadi_assert_message(dae_.getNumInputs()==DAE_NUM_IN,"Wrong number of inputs for the DAE callback function");
    casadi_assert_message(dae_.getNumOutputs()==DAE_NUM_OUT,"Wrong number of outputs for the DAE callback function");
  
    // Get dimensions
    nx_ = dae_.input(DAE_X).size();
    nz_ = dae_.input(DAE_Z).size();
    np_ = dae_.input(DAE_P).size();
    nq_ = dae_.output(DAE_QUAD).size();
    
    // Initialize the functions
    casadi_assert(!f_.isNull());
  
    // Initialize, get and assert dimensions of the forward integration
    if(!f_.isInit()) f_.init();
    casadi_assert_message(f_.getNumInputs()==DAE_NUM_IN,"Wrong number of inputs for the DAE callback function");
    casadi_assert_message(f_.getNumOutputs()==DAE_NUM_OUT,"Wrong number of outputs for the DAE callback function");
    casadi_assert_message(f_.input(DAE_X).dense(),"State vector must be dense in the DAE callback function");
    casadi_assert_message(f_.output(DAE_ODE).dense(),"Right hand side vector must be dense in the DAE callback function");
    nfx_ = f_.input(DAE_X).numel();
    nfz_ = f_.input(DAE_Z).numel();
    nfq_ = f_.output(DAE_QUAD).numel();
    nfp_  = f_.input(DAE_P).numel();
    casadi_assert_message(f_.output(DAE_ODE).numel()==nfx_,"Inconsistent dimensions. Expecting DAE_ODE output of size " << nfx_ << ", but got " << f_.output(DAE_ODE).numel() << " instead.");
    casadi_assert_message(f_.output(DAE_ALG).numel()==nfz_,"Inconsistent dimensions. Expecting DAE_ALG output of size " << nfz_ << ", but got " << f_.output(DAE_ALG).numel() << " instead.");
  
    // Initialize, get and assert dimensions of the backwards integration
    if(g_.isNull()){
      // No backwards integration
      nrx_ = nrz_ = nrq_ = nrp_ = 0;
    } else {
      if(!g_.isInit()) g_.init();
      casadi_assert_message(g_.getNumInputs()==RDAE_NUM_IN,"Wrong number of inputs for the backwards DAE callback function");
      casadi_assert_message(g_.getNumOutputs()==RDAE_NUM_OUT,"Wrong number of outputs for the backwards DAE callback function");
      nrx_ = g_.input(RDAE_RX).numel();
      nrz_ = g_.input(RDAE_RZ).numel();
      nrp_ = g_.input(RDAE_RP).numel();
      nrq_ = g_.output(RDAE_QUAD).numel();
      casadi_assert_message(g_.input(RDAE_P).numel()==nfp_,"Inconsistent dimensions. Expecting RDAE_P input of size " << nfp_ << ", but got " << g_.input(RDAE_P).numel() << " instead.");
      casadi_assert_message(g_.input(RDAE_X).numel()==nfx_,"Inconsistent dimensions. Expecting RDAE_X input of size " << nfx_ << ", but got " << g_.input(RDAE_X).numel() << " instead.");
      casadi_assert_message(g_.input(RDAE_Z).numel()==nfz_,"Inconsistent dimensions. Expecting RDAE_Z input of size " << nfz_ << ", but got " << g_.input(RDAE_Z).numel() << " instead.");
      casadi_assert_message(g_.output(RDAE_ODE).numel()==nrx_,"Inconsistent dimensions. Expecting RDAE_ODE output of size " << nrx_ << ", but got " << g_.output(RDAE_ODE).numel() << " instead.");
      casadi_assert_message(g_.output(RDAE_ALG).numel()==nrz_,"Inconsistent dimensions. Expecting RDAE_ALG input of size " << nrz_ << ", but got " << g_.output(RDAE_ALG).numel() << " instead.");
    }
    casadi_assert(nfx_ == (1+nfwd_)*nx_);
    casadi_assert(nrx_ == nadj_*nx_);
  
    // Inputs
    setNumInputs(NEW_INTEGRATOR_NUM_IN*(1+nfwd_) + NEW_INTEGRATOR_NUM_OUT*nadj_);
    int ind=0;
    for(int d=-1; d<nfwd_; ++d){
      input(ind++) = DMatrix(dae_.input(DAE_X).sparsity(),0);
      input(ind++) = DMatrix(dae_.input(DAE_P).sparsity(),0);
    }
    for(int d=0; d<nadj_; ++d){
      input(ind++) = DMatrix(dae_.output(DAE_ODE).sparsity(),0);
      input(ind++) = DMatrix(dae_.output(DAE_QUAD).sparsity(),0);
    }
    casadi_assert(ind == getNumInputs());

    // Outputs
    setNumOutputs(NEW_INTEGRATOR_NUM_OUT*(1+nfwd_) + NEW_INTEGRATOR_NUM_IN*nadj_);
    ind=0;
    for(int d=-1; d<nfwd_; ++d){
      output(ind++) = DMatrix(dae_.output(DAE_ODE).sparsity(),0);
      output(ind++) = DMatrix(dae_.output(DAE_QUAD).sparsity(),0);
    }
    for(int d=0; d<nadj_; ++d){
      output(ind++) = DMatrix(dae_.input(DAE_X).sparsity(),0);
      output(ind++) = DMatrix(dae_.input(DAE_P).sparsity(),0);
    }
    casadi_assert(ind == getNumOutputs());
  
    // Call the base class method
    FXInternal::init();

    {
      std::stringstream ss;
      ss << "Integrator dimensions: nx=" << nfx_ << ", nz="<< nfz_ << ", nq=" << nfq_ << ", np=" << nfp_;
      log("IntegratorInternal::init",ss.str());
    }
  
    // read options
    t0_ = getOption("t0");
    tf_ = getOption("tf");
    fwd_via_sct_ = getOption("fwd_via_sct");
    adj_via_sct_ = getOption("adj_via_sct");
  }

  void IntegratorInternal::deepCopyMembers(std::map<SharedObjectNode*,SharedObject>& already_copied){
    FXInternal::deepCopyMembers(already_copied);
    f_ = deepcopy(f_,already_copied);
    g_ = deepcopy(g_,already_copied);
  }

  std::pair<FX,FX> IntegratorInternal::getAugmented(int nfwd, int nadj){
    log("IntegratorInternal::getAugmented","call");
    if(is_a<SXFunction>(f_)){
      casadi_assert_message(g_.isNull() || is_a<SXFunction>(g_), "Currently, g_ must be of the same type as f_");
      return getAugmentedGen<SXMatrix,SXFunction>(nfwd,nadj);
    } else if(is_a<MXFunction>(f_)){
      casadi_assert_message(g_.isNull() || is_a<MXFunction>(g_), "Currently, g_ must be of the same type as f_");
      return getAugmentedGen<MX,MXFunction>(nfwd,nadj);
    } else {
      throw CasadiException("Currently, f_ must be either SXFunction or MXFunction");
    }
  }
  
  template<class Mat,class XFunc>
  std::pair<FX,FX> IntegratorInternal::getAugmentedGen(int nfwd, int nadj){
  
    log("IntegratorInternal::getAugmentedGen","begin");
  
    // Get derivatived type
    XFunc f = shared_cast<XFunc>(f_);
    XFunc g = shared_cast<XFunc>(g_);
  
    // Take apart forward problem
    vector<Mat> dae_in = f.inputExpr();
    vector<Mat> dae_out = f.outputExpr();
    casadi_assert(dae_in.size()==DAE_NUM_IN);
    casadi_assert(dae_out.size()==DAE_NUM_OUT);
    Mat x = dae_in[DAE_X];
    Mat z = dae_in[DAE_Z];
    Mat p = dae_in[DAE_P];
    Mat t = dae_in[DAE_T];
    Mat ode = dae_out[DAE_ODE];
    Mat alg = dae_out[DAE_ALG];
    Mat quad = dae_out[DAE_QUAD];
  
    // Take apart the backwards problem
    vector<Mat> rdae_in(RDAE_NUM_IN), rdae_out(RDAE_NUM_OUT);
    if(!g.isNull()){
      rdae_in = g.inputExpr();
      rdae_out = g.outputExpr();
      // TODO: Assert that rdae_in[RDAE_X]==x, rdae_in[RDAE_Z]==z, rdae_in[RDAE_P]==p
    } else {
      rdae_in[RDAE_X]=x;
      rdae_in[RDAE_Z]=z;
      rdae_in[RDAE_P]=p;
      rdae_in[RDAE_T]=t;
    }
    Mat rx = rdae_in[RDAE_RX];
    Mat rz = rdae_in[RDAE_RZ];
    Mat rp = rdae_in[RDAE_RP];
    Mat rode = rdae_out[RDAE_ODE];
    Mat ralg = rdae_out[RDAE_ALG];
    Mat rquad = rdae_out[RDAE_QUAD];
  
    // Function evaluating f and g
    vector<Mat> fg_out(DAE_NUM_OUT+RDAE_NUM_OUT);
    copy(dae_out.begin(),dae_out.end(),fg_out.begin());
    copy(rdae_out.begin(),rdae_out.end(),fg_out.begin()+DAE_NUM_OUT);
    XFunc fg(rdae_in,fg_out);
    fg.init();
  
    // Allocate forward sensitivities
    vector<Mat> fwd_x = Mat::sym("fwd_x",x.sparsity(),nfwd);
    vector<Mat> fwd_z = Mat::sym("fwd_z",z.sparsity(),nfwd);
    vector<Mat> fwd_p = Mat::sym("fwd_p",p.sparsity(),nfwd);
    vector<Mat> fwd_rx = Mat::sym("fwd_rx",rx.sparsity(),nfwd);
    vector<Mat> fwd_rz = Mat::sym("fwd_rz",rz.sparsity(),nfwd);
    vector<Mat> fwd_rp = Mat::sym("fwd_rp",rp.sparsity(),nfwd);

    // Allocate adjoint sensitivities
    vector<Mat> adj_ode = Mat::sym("adj_ode",ode.sparsity(),nadj);
    vector<Mat> adj_alg = Mat::sym("adj_alg",alg.sparsity(),nadj);
    vector<Mat> adj_quad = Mat::sym("adj_quad",quad.sparsity(),nadj);
    vector<Mat> adj_rode = Mat::sym("adj_rode",rode.sparsity(),nadj);
    vector<Mat> adj_ralg = Mat::sym("adj_ralg",ralg.sparsity(),nadj);
    vector<Mat> adj_rquad = Mat::sym("adj_rquad",rquad.sparsity(),nadj);
    
    // Forward seeds
    vector<vector<Mat> > fseed(nfwd,vector<Mat>(RDAE_NUM_IN));
    for(int dir=0; dir<nfwd; ++dir){
      fseed[dir][RDAE_X] = fwd_x[dir];
      fseed[dir][RDAE_Z] = fwd_z[dir];
      fseed[dir][RDAE_P] = fwd_p[dir];
      if(!t.isNull()) fseed[dir][RDAE_T] = Mat(t.sparsity());
      fseed[dir][RDAE_RX] = fwd_rx[dir];
      fseed[dir][RDAE_RZ] = fwd_rz[dir];
      fseed[dir][RDAE_RP] = fwd_rp[dir];
    }

    // Adjoint seeds
    vector<vector<Mat> > aseed(nadj,vector<Mat>(DAE_NUM_OUT+RDAE_NUM_OUT));
    for(int dir=0; dir<nadj; ++dir){
      aseed[dir][DAE_ODE] = adj_ode[dir];
      aseed[dir][DAE_ALG] = adj_alg[dir];
      aseed[dir][DAE_QUAD] = adj_quad[dir];
    
      aseed[dir][DAE_NUM_OUT+RDAE_ODE] = adj_rode[dir];
      aseed[dir][DAE_NUM_OUT+RDAE_ALG] = adj_ralg[dir];
      aseed[dir][DAE_NUM_OUT+RDAE_QUAD] = adj_rquad[dir];
    }
  
    // Calculate forward and adjoint sensitivities
    vector<vector<Mat> > fsens(fseed.size(),fg_out);
    vector<vector<Mat> > asens(aseed.size(),rdae_in);
    fg.eval(rdae_in,fg_out,fseed,fsens,aseed,asens);
  
    // Augment differential state
    x.append(vertcat(fwd_x));
    x.append(vertcat(adj_rode));
  
    // Augment algebraic state
    z.append(vertcat(fwd_z));
    z.append(vertcat(adj_ralg));
  
    // Augment parameter vector
    p.append(vertcat(fwd_p));
    p.append(vertcat(adj_rquad));
  
    // Augment backward differential state
    rx.append(vertcat(fwd_rx));
    rx.append(vertcat(adj_ode));
  
    // Augment backward algebraic state
    rz.append(vertcat(fwd_rz));
    rz.append(vertcat(adj_alg));
  
    // Augment backwards parameter vector
    rp.append(vertcat(fwd_rp));
    rp.append(vertcat(adj_quad));
  
    // Augment forward sensitivity equations to the DAE
    for(int dir=0; dir<nfwd; ++dir){
      ode.append(fsens[dir][DAE_ODE]);
      alg.append(fsens[dir][DAE_ALG]);
      quad.append(fsens[dir][DAE_QUAD]);

      rode.append(fsens[dir][DAE_NUM_OUT+RDAE_ODE]);
      ralg.append(fsens[dir][DAE_NUM_OUT+RDAE_ALG]);
      rquad.append(fsens[dir][DAE_NUM_OUT+RDAE_QUAD]);
    }
  
    // Augment backward sensitivity equations to the DAE
    for(int dir=0; dir<nadj; ++dir){
      rode.append(asens[dir][RDAE_X]);
      ralg.append(asens[dir][RDAE_Z]);
      rquad.append(asens[dir][RDAE_P]);
      ode.append(asens[dir][RDAE_RX]);
      alg.append(asens[dir][RDAE_RZ]);
      quad.append(asens[dir][RDAE_RP]);
    }
  
    // Make sure that the augmented problem is dense
    makeDense(ode);
    makeDense(alg);
    makeDense(quad);
    makeDense(rode);
    makeDense(ralg);
    makeDense(rquad);
  
    // Update the forward problem inputs ...
    dae_in[DAE_X] = x;
    dae_in[DAE_Z] = z;
    dae_in[DAE_P] = p;
    dae_in[DAE_T] = t;

    // ... and outputs
    dae_out[DAE_ODE] = ode;
    dae_out[DAE_ALG] = alg;
    dae_out[DAE_QUAD] = quad;
  
    // Update the backward problem inputs ...
    rdae_in[RDAE_RX] = rx;
    rdae_in[RDAE_RZ] = rz;
    rdae_in[RDAE_RP] = rp;
    rdae_in[RDAE_X] = x;
    rdae_in[RDAE_Z] = z;
    rdae_in[RDAE_P] = p;
    rdae_in[RDAE_T] = t;
  
    // ... and outputs
    rdae_out[RDAE_ODE] = rode;
    rdae_out[RDAE_ALG] = ralg;
    rdae_out[RDAE_QUAD] = rquad;
  
    // Create functions for the augmented problems
    XFunc f_aug(dae_in,dae_out);
    XFunc g_aug(rdae_in,rdae_out);

    f_aug.init();
  
    casadi_assert_message(f_aug.getFree().size()==0,"IntegratorInternal::getDerivative: Found free variables " << f_aug.getFree() << " while constructing augmented dae. Make sure that gx, gz and gq have a linear dependency on rx, rz and rp. This is a restriction of the implementation.");
  
    // Workaround, delete g_aug if its empty
    if(g.isNull() && nadj==0) g_aug = XFunc();
  
    log("IntegratorInternal::getAugmentedGen","end");
    
    return pair<FX,FX>(f_aug,g_aug);
  }

  void IntegratorInternal::spEvaluate(bool fwd){
    log("IntegratorInternal::spEvaluate","begin");
    /**  This is a bit better than the FXInternal implementation: XF and QF never depend on RX0 and RP, 
     *   i.e. the worst-case structure of the Jacobian is:
     *        x0  p rx0 rp
     *        --------------
     *   xf  | x  x        |
     *   qf  | x  x        |
     *  rxf  | x  x  x  x  |
     *  rqf  | x  x  x  x  |
     *        --------------
     * 
     *  An even better structure of the Jacobian can be obtained by propagating sparsity through the callback functions.
     */
  
    // Variable which depends on all states and parameters
    bvec_t all_depend(0);
  
    if(fwd){
    
      // Have dependency on anything in x0 or p
      for(int k=0; k<2; ++k){
        int iind = k==0 ? INTEGRATOR_X0 : INTEGRATOR_P;
        const DMatrix& m = inputNoCheck(iind);
        const bvec_t* v = reinterpret_cast<const bvec_t*>(m.ptr());
        for(int i=0; i<m.size(); ++i){
          all_depend |= v[i];
        }
      }
    
      // Propagate to xf and qf (that only depend on x0 and p)
      for(int k=0; k<2; ++k){
        int oind = k==0 ? INTEGRATOR_XF : INTEGRATOR_QF;
        DMatrix& m = outputNoCheck(oind);
        bvec_t* v = reinterpret_cast<bvec_t*>(m.ptr());
        for(int i=0; i<m.size(); ++i){
          v[i] = all_depend;
        }
      }
    
      // Add dependency on rx0 or rp
      for(int k=0; k<2; ++k){
        int iind = k==0 ? INTEGRATOR_RX0 : INTEGRATOR_RP;
        const DMatrix& m = inputNoCheck(iind);
        const bvec_t* v = reinterpret_cast<const bvec_t*>(m.ptr());
        for(int i=0; i<m.size(); ++i){
          all_depend |= v[i];
        }
      }
    
      // Propagate to rxf and rqf
      for(int k=0; k<2; ++k){
        int oind = k==0 ? INTEGRATOR_RXF : INTEGRATOR_RQF;
        DMatrix& m = outputNoCheck(oind);
        bvec_t* v = reinterpret_cast<bvec_t*>(m.ptr());
        for(int i=0; i<m.size(); ++i){
          v[i] = all_depend;
        }
      }
    
    } else {
    
      // First find out what influences only rxf and rqf
      for(int k=0; k<2; ++k){
        int oind = k==0 ? INTEGRATOR_RXF : INTEGRATOR_RQF;
        const DMatrix& m = outputNoCheck(oind);
        const bvec_t* v = get_bvec_t(m.data());
        for(int i=0; i<m.size(); ++i){
          all_depend |= v[i];
        }
      }
    
      // Propagate to rx0 and rp
      for(int k=0; k<2; ++k){
        int iind = k==0 ? INTEGRATOR_RX0 : INTEGRATOR_RP;
        DMatrix& m = inputNoCheck(iind);
        bvec_t* v = get_bvec_t(m.data());
        for(int i=0; i<m.size(); ++i){
          v[i] = all_depend;
        }
      }
    
      // Add dependencies to xf and qf
      for(int k=0; k<2; ++k){
        int oind = k==0 ? INTEGRATOR_XF : INTEGRATOR_QF;
        const DMatrix& m = outputNoCheck(oind);
        const bvec_t* v = get_bvec_t(m.data());
        for(int i=0; i<m.size(); ++i){
          all_depend |= v[i];
        }
      }
    
      // Propagate to x0 and p
      for(int k=0; k<2; ++k){
        int iind = k==0 ? INTEGRATOR_X0 : INTEGRATOR_P;
        DMatrix& m = inputNoCheck(iind);
        bvec_t* v = get_bvec_t(m.data());
        for(int i=0; i<m.size(); ++i){
          v[i] = all_depend;
        }
      }
    }
    log("IntegratorInternal::spEvaluate","end");
  }

  FX IntegratorInternal::getDerivative(int nfwd, int nadj){

    log("IntegratorInternal::getDerivative","begin");
    // Generate augmented DAE
    std::pair<FX,FX> aug_dae = getAugmented(nfwd,nadj);
  
    // Create integrator for augmented DAE
    Integrator integrator;
    integrator.assignNode(create(dae_,(1+nfwd_)*(1+nfwd) + nadj_*nadj - 1, (1+nfwd_)*nadj + nadj_*(1+nfwd)));
    integrator->setF(aug_dae.first);
    integrator->setG(aug_dae.second);
  
    // Copy options
    integrator.setOption(dictionary());
  
    // Pass down specific options if provided
    if (hasSetOption("augmented_options"))
      integrator.setOption(getOption("augmented_options"));
  
    // Initialize the integrator since we will call it below
    integrator.init();
    vector<MX> integrator_in = integrator.symbolicInput();
    vector<MX> integrator_out = integrator.call(integrator_in);

    vector<MX> ret_in(integrator_in.size());
    vector<MX> ret_out(integrator_out.size());
    vector<MX>::const_iterator integrator_in_it = integrator_in.begin();
    vector<MX>::const_iterator integrator_out_it = integrator_out.begin();

    for(int d1=-1; d1<nfwd; ++d1){
      for(int d2=-1; d2<nfwd_; ++d2){
        ret_in[getNumInputs()*(1+d1) + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_X0] = *integrator_in_it++;
        ret_in[getNumInputs()*(1+d1) + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_P] = *integrator_in_it++;

        ret_out[getNumOutputs()*(1+d1) + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_XF] = *integrator_out_it++;
        ret_out[getNumOutputs()*(1+d1) + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_QF] = *integrator_out_it++;
      }
    }

    for(int d1=0; d1<nadj; ++d1){
      for(int d2=0; d2<nadj_; ++d2){
        ret_in[getNumInputs()*(1+nfwd) + getNumOutputs()*d1 + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_X0] = *integrator_in_it++;
        ret_in[getNumInputs()*(1+nfwd) + getNumOutputs()*d1 + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_P] = *integrator_in_it++;

        ret_out[getNumOutputs()*(1+nfwd) + getNumInputs()*d1 + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_XF] = *integrator_out_it++;
        ret_out[getNumOutputs()*(1+nfwd) + getNumInputs()*d1 + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_QF] = *integrator_out_it++;
      }
    }

    for(int d1=0; d1<nadj; ++d1){
      for(int d2=-1; d2<nfwd_; ++d2){
        ret_in[getNumInputs()*(1+nfwd) + getNumOutputs()*d1 + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_XF] = *integrator_in_it++;
        ret_in[getNumInputs()*(1+nfwd) + getNumOutputs()*d1 + NEW_INTEGRATOR_NUM_OUT*(1+d2) + NEW_INTEGRATOR_QF] = *integrator_in_it++;

        ret_out[getNumOutputs()*(1+nfwd) + getNumInputs()*d1 + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_X0] = *integrator_out_it++;
        ret_out[getNumOutputs()*(1+nfwd) + getNumInputs()*d1 + NEW_INTEGRATOR_NUM_IN*(1+d2) + NEW_INTEGRATOR_P] = *integrator_out_it++;
      }
    }

    for(int d1=-1; d1<nfwd; ++d1){
      for(int d2=0; d2<nadj_; ++d2){
        ret_in[getNumInputs()*(1+d1) + NEW_INTEGRATOR_NUM_IN*(1+nfwd_) + NEW_INTEGRATOR_NUM_OUT*d2 + NEW_INTEGRATOR_XF] = *integrator_in_it++;
        ret_in[getNumInputs()*(1+d1) + NEW_INTEGRATOR_NUM_IN*(1+nfwd_) + NEW_INTEGRATOR_NUM_OUT*d2 + NEW_INTEGRATOR_QF] = *integrator_in_it++;

        ret_out[getNumOutputs()*(1+d1) + NEW_INTEGRATOR_NUM_OUT*(1+nfwd_) + NEW_INTEGRATOR_NUM_IN*d2 + NEW_INTEGRATOR_X0] = *integrator_out_it++;
        ret_out[getNumOutputs()*(1+d1) + NEW_INTEGRATOR_NUM_OUT*(1+nfwd_) + NEW_INTEGRATOR_NUM_IN*d2 + NEW_INTEGRATOR_P] = *integrator_out_it++;
      }
    }

    return MXFunction(ret_in,ret_out);
  }

  FX IntegratorInternal::getJacobian(int iind, int oind, bool compact, bool symmetric){
    vector<MX> arg = symbolicInput();
    vector<MX> res = shared_from_this<FX>().call(arg);
    MXFunction f(arg,res);
    f.setOption("ad_mode","forward");
    f.setOption("numeric_jacobian", false);
    f.init();
    return f.jacobian(iind,oind,compact,symmetric);
  }

  void IntegratorInternal::reset(int nsens, int nsensB, int nsensB_store){
    log("IntegratorInternal::reset","begin");
    // Make sure that the numbers are consistent
    casadi_assert_message(nsens<=nfdir_,"Too many sensitivities going forward");
    casadi_assert_message(nsensB<=nfdir_,"Too many sensitivities going backward");
    casadi_assert_message(nsensB_store<=nsens,"Too many sensitivities stored going forward");
    casadi_assert_message(nsensB_store<=nsensB,"Too many sensitivities stored going backward");
  
    nsens_ = nsens;
    nsensB_ = nsensB;
    nsensB_store_ = nsensB_store;
  
    // Initialize output (relevant for integration with a zero advance time )
    copy(input(INTEGRATOR_X0).begin(),input(INTEGRATOR_X0).end(),output(INTEGRATOR_XF).begin());
    for(int i=0; i<nfdir_; ++i)
      copy(fwdSeed(INTEGRATOR_X0,i).begin(),fwdSeed(INTEGRATOR_X0,i).end(),fwdSens(INTEGRATOR_XF,i).begin());

    log("IntegratorInternal::reset","end");
  }


} // namespace CasADi


