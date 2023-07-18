# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:00:29 2019

@author: Armin Galetzka

Python implementation for:

An hp-adaptive multi-element stochastic collocation method for
surrogate modeling with information re-use

abstract:
This paper introduces an hp-adaptive multi-element stochastic collocation method,
which additionally allows to re-use existing model evaluations during either h- or
p-refinement. The collocation method is based on weighted Leja nodes. After h-
refinement, local interpolations are stabilized by adding and sorting Leja nodes on
each newly created sub-element in a hierarchical manner. For p-refinement, the local
polynomial approximations are based on total-degree or dimension-adaptive bases.
The method is applied in the context of forward and inverse uncertainty quantification
to handle non-smooth or strongly localised response surfaces. The performance of the
proposed method is assessed in several test cases, also in comparison to competing
methods.

Note, this implementation only considers TD bases.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from leja import seq_lj_1d, lejaorder
from lagrange1d import Hierarchical1d
from interpolation import interpolate_single, interpolate_multiple
from td_multiindex_set import td_multiindex_set
from pce_tools import transform_multi_index_set, get_design_matrix, get_dist_ot, get_mv_basis_ot, map_data, compute_moments
import scipy.linalg as scl
from scipy.integrate import quad
from distributions import Uniform, Normal, TruncNormal, Beta


class Tree():
    """
    Class for Tree nodes. Each node is also refered to as an element of the 
    parameter domain decomposition
    """

    # class variables
    id_count = 0             # id of tree node
    fcalls = 0               # number of function calls
    fcalls_lost = 0          # lost function calls (through h-refinement)
    fcalls_possible_saving = 0  # possible saving through h-refinement
    verbose = False           # enable talkative version of algorithm
    original_dist = None       # save original distribution on first node
    cv_in = None              # nodes for validation samples
    cv_out = None             # function evaluations for validation samples

    # general parameters
    parameter = {}
    parameter['use_exact_error'] = False        # use the exact error for refinement decision
    parameter['nodes'] = 'leja'                 # 'leja'
    parameter['href_split_marker'] = 'median'   # 'median', 'mean'
    parameter['marking_strategy'] = 'absolute'  # 'absolute', 'maximum_strategy', 'fixed_energy_fraction'
    parameter['basis'] = 'TD'                   # 'TD'

    # leja parameters
    parameter_leja = {}
    parameter_leja['error_ind_with_admissible'] = False
    parameter_leja['reuse_Leja_nodes_href'] = True
    parameter_leja['sort_Leja_nodes'] = True

    # parameters for easy to access 
    nodes = 'leja'                            # node type (only Leja implemented)
    reuse_Leja_nodes_href = True              # reuse Leja nodes after h-refinement
    sort_Leja_nodes = True                    # sort Leja nodes after h-refinement
    basis = 'TD'                              # 'TD'
    href_split_marker = 'median'              # 'median', 'mean'
    marking_strategy = 'maximum_strategy'     # 'absolute', 'maximum_strategy', 'fixed_energy_fraction'

    def __init__(self, f, dist, data=None, knots_per_dim=None, p_max=8, discontinuous=False, dim_refined=None, id_in=1):
        """
        Parameters
        ----------
        f               : black box funtion to be approximated
        dist            : distribution for the RV
        data            : optional, initial data vector R^(N+1), last column is for function evals
        knots_per_dim   : optional, Leja knots per dimension
        p_max           : optional, polynomial degree for Leja/gPC basis
        discontinuous   : optional, objective function is assumed to have discontinuities
        dim_refined     : optional, dimension that has been refined to construct this leaf
        id_in           : optional, id of new leaf
        """

        self._write_parameters()

        self.interp_dict = None
        self.mark = False
        self.N = len(dist)
        self.dim_refined = dim_refined
        self.id = id_in

        # first node
        if self.id == 1:
            self.k = 0  # refinment counter
            if data is not None:
                Tree.fcalls = data.shape[0]
            else: 
                Tree.fcalls = 0
            Tree.original_dist = dist

        # init structures
        self.knots_per_dim = knots_per_dim
        if self.knots_per_dim is None:
            self.knots_per_dim = {}
            for n in range(self.N):
                self.knots_per_dim[n] = np.array([]).reshape(-1, 1)

        self.data_init = data
        self.data = data
        if self.data is None:
            self.data = np.array([]).reshape(-1, 1)
            self.data_init = np.array([]).reshape(-1, 1)

        # assign parameters
        self.bounds = [dist[i].bounds for i in range(self.N)]
        self.p_max = p_max
        self.left = None
        self.right = None
        self.f = f
        self.dist = dist
        self.approximated = False
        self.discontinuous = discontinuous
        self.refinable = True
        self.Jk = np.prod([[self.dist[i].get_normalization_const()] for i in range(
            self.N)])  # calculate probaility mass of jpdf (after refinment < 1 !)
        self.lebesgue_constant = None
        self.id = id_in
        
        # init validation error on element
        self.cv_error_max = None
        self.cv_error_mean = None
        self.cv_error_rms = None

        Tree.id_count += 1

    def _refine(self, theta2, thetap):
        """ Refines a leaf.
        """

        # decide whether p- or h- refinment is required
        _, m = self._get_p_convergence_rate()
        self.p_indicator = np.exp(-m)

        if self.p_indicator <= thetap:
            # p-refinment
            self.p_max += 1
            self.approximated = False
            print('|' + str(self.id) + '|: p-ref: ' +
                  str(self.p_max-1) + '->' + str(self.p_max))

        else:
            # h-refinment
            parameter_to_refine = self._get_dimension_to_refine(theta2)
            self._h_refine_multiple_leaves(parameter_to_refine, [self])

    def _h_refine_multiple_leaves(self, parameter_to_refine, leaves_to_refine):
        new_leaves = []
        while len(parameter_to_refine) > 0:
            i = parameter_to_refine.pop(0)
            leaves_to_refine.extend(new_leaves)
            while len(leaves_to_refine) > 0:
                leaf = leaves_to_refine.pop(0)
                left, right = self._h_refine(i, leaf)
                if left is not None and right is not None:
                    new_leaves.append(left)
                    new_leaves.append(right)

    def _h_refine(self, i, leaf):

        # save current bounds
        left_bounds = leaf.bounds.copy()
        right_bounds = leaf.bounds.copy()

        # get new bounds
        # check if distribution is uniform
        if self.dist[i].type == 'uniform':
            half_domain = np.abs((leaf.bounds[i][1]-leaf.bounds[i][0])/2.)
            # split domain in half and create new bounds (median = mean!)
            left_bounds[i] = [leaf.bounds[i][0], leaf.bounds[i][1]-half_domain]
            right_bounds[i] = [leaf.bounds[i][1] -
                               half_domain, leaf.bounds[i][1]]
        else:
            if self.href_split_marker == 'mean':
                half_domain = leaf.dist[i].get_mean()
            elif self.href_split_marker == 'median':
                half_domain = leaf.dist[i].get_median()
            else:
                raise RuntimeError(
                    'Splitting marker must be \"median\" or \"mean\"')

            left_bounds[i] = [leaf.bounds[i][0], half_domain]
            right_bounds[i] = [half_domain, leaf.bounds[i][1]]

        # divide data for left and right child
        data_left = leaf.data[leaf.data[:, i] <= left_bounds[i][1], :]
        if self.discontinuous:
            data_right = leaf.data[leaf.data[:, i] > left_bounds[i][1], :]
        else:
            data_right = leaf.data[leaf.data[:, i] >= left_bounds[i][1], :]
        # divide univariate knots per dim
        knots_per_dim_left = {}
        knots_per_dim_right = {}
        for n in range(self.N):
            if i == n:
                knots_per_dim_left[n] = leaf.knots_per_dim[n][leaf.knots_per_dim[n]
                                                              <= left_bounds[i][1]]
                knots_per_dim_right[n] = leaf.knots_per_dim[n][leaf.knots_per_dim[n]
                                                               >= left_bounds[i][1]]
                # if objective function is discontinuous, remove nodes on the boundary
                if self.discontinuous:
                    knots_per_dim_left[n] = leaf.knots_per_dim[n][leaf.knots_per_dim[n]
                                                                  <= left_bounds[i][1]]
                    knots_per_dim_right[n] = leaf.knots_per_dim[n][leaf.knots_per_dim[n]
                                                                   > left_bounds[i][1]]
            else:
                knots_per_dim_left[n] = leaf.knots_per_dim[n]
                knots_per_dim_right[n] = leaf.knots_per_dim[n]

        if Tree.verbose:
            print('ed size:', leaf.data[:, :-1].shape[0])
            print('bounds for element: ' + str(leaf.bounds) + ' --> ' +
                  str(left_bounds) + ' & ' + str(right_bounds))
            print('--------------------')
            print('')

        # copy ALL old distributions
        dist_left = [leaf.dist[n].get_copy()
                     for n in range(self.N)]  # copy.deepcopy(leaf.dist)
        dist_right = [leaf.dist[n].get_copy()
                      for n in range(self.N)]  # copy.deepcopy(leaf.dist)
        # create new distribution for the dimension that is refined
        if dist_left[i].type == 'uniform':
            dist_left[i] = Uniform([leaf.dist[i].a, leaf.dist[i].b])
            dist_left[i].set_support(left_bounds[i])
            dist_right[i] = Uniform([leaf.dist[i].a, leaf.dist[i].b])
            dist_right[i].set_support(right_bounds[i])
        elif dist_left[i].type == 'normal':
            dist_left[i] = Normal(leaf.dist[i].mean, leaf.dist[i].var, [
                                  leaf.dist[i].a, leaf.dist[i].b])
            dist_left[i].set_support(left_bounds[i])
            dist_right[i] = Normal(leaf.dist[i].mean, leaf.dist[i].var, [
                                   leaf.dist[i].a, leaf.dist[i].b])
            dist_right[i].set_support(right_bounds[i])
        elif dist_left[i].type == 'beta':
            dist_left[i] = Beta(leaf.dist[i].alpha, leaf.dist[i].beta, leaf.dist[i].mean, leaf.dist[i].var, [
                                leaf.dist[i].a, leaf.dist[i].b])
            dist_left[i].set_support(left_bounds[i])
            dist_right[i] = Beta(leaf.dist[i].alpha, leaf.dist[i].beta, leaf.dist[i].mean, leaf.dist[i].var, [
                                 leaf.dist[i].a, leaf.dist[i].b])
            dist_right[i].set_support(right_bounds[i])
        elif dist_left[i].type == 'truncated_normal':
            def f(x):
                return Tree.original_dist[i].pdf(x)
            norm_const = quad(f, left_bounds[i][0], left_bounds[i][1])[0]
            dist_left[i] = TruncNormal(
                leaf.dist[i].mean, leaf.dist[i].var, left_bounds[i])
            dist_left[i].set_support(left_bounds[i])
            dist_left[i].set_normalization_const(norm_const)

            def f(x):
                return Tree.original_dist[i].pdf(x)
            norm_const = quad(f, right_bounds[i][0], right_bounds[i][1])[0]
            dist_right[i] = TruncNormal(
                leaf.dist[i].mean, leaf.dist[i].var, right_bounds[i])
            dist_right[i].set_support(right_bounds[i])
            dist_right[i].set_normalization_const(norm_const)
        else:
            RuntimeError('Distribution not (yet) implemented!')

        # create new leaves
        pmax_left = leaf.p_max
        pmax_right = leaf.p_max

        leaf.left = Tree(leaf.f, dist_left, data_left, knots_per_dim_left,
                         pmax_left, leaf.discontinuous, i, id_in=Tree.id_count+1)
        leaf.right = Tree(leaf.f, dist_right, data_right, knots_per_dim_right,
                          pmax_right, leaf.discontinuous, i, id_in=Tree.id_count+1)

        return leaf.left, leaf.right

    def _get_dimension_to_refine(self, theta2):
        S = []
        for n in range(self.N):
            true_false = np.where(
                np.sum(np.delete(np.array(self.idx_act_gpc), n, axis=1), axis=1) == 0)[0]
            S.append(
                np.mean(np.asarray(self.pce_coeff_sur)[true_false[1:]]**2))
        dim_to_refine = [np.argmax(S)]

        print('|' + str(self.id) + '|: h-ref, dim: ' + str(dim_to_refine))
        return dim_to_refine

    def _mark_leaves(self, theta1, leaves):
        error_indicators = [leaf.local_error_ind for leaf in leaves]
        Jkz = [leaf.Jk for leaf in leaves]
        if self.marking_strategy == 'absolute':
            for leaf in leaves:
                leaf.mark = leaf.local_error_ind**0.5*leaf.Jk >= theta1  # Wan, Karniadakis 2005
                leaf.refinable = leaf.mark
        elif self.marking_strategy == 'maximum_strategy':
            for leaf in leaves:
                leaf.mark = leaf.local_error_ind**0.5 * \
                    leaf.Jk >= (
                        1.-theta1)*np.max(np.array(error_indicators)**0.5*np.array(Jkz))
                leaf.refinable = leaf.mark
        elif self.marking_strategy == 'fixed_energy_fraction':
            idx_sort = np.argsort(error_indicators).tolist()
            idx_sort.reverse()
            idx_sort = np.array(idx_sort)
            mark_sum = []
            total_error = np.sum(
                [(leaf.local_error_ind*leaf.Jk)**2 for leaf in leaves])
            for i in idx_sort:
                leaves[i].mark = True
                leaves[i].refinable = leaves[i].mark
                if np.sum(mark_sum) >= theta1**2*total_error:
                    leaves[i].mark = False
                    leaves[i].refinable = leaves[i].mark
                mark_sum.append((leaves[i].local_error_ind*leaves[i].Jk)**2)

    def _get_local_error_indicator(self):
        if self.use_exact_error:
            local_cv_in, local_cv_out = self._sort_data_to_bounds(
                Tree.cv_in, self.bounds, data_out=Tree.cv_out)
            local_cv_sur = self._evaluate(local_cv_in)
            errors = np.abs(local_cv_sur-local_cv_out)
            err = np.sqrt(np.sum(errors**2)/local_cv_in.shape[0])
        else:
            p_max = np.max(np.sum(self.idx_act_gpc, axis=1))
            idx = np.where(np.sum(self.idx_act_gpc, axis=1) == p_max)
            highest_order_variances = np.asarray(self.pce_coeff_sur)[idx]
            if self.local_variance_PCE != 0 and self.local_variance_PCE > np.finfo(float).eps:
                err = (np.sum((highest_order_variances)**2) /
                        self.local_variance_PCE)
                if np.isnan(err):
                    err = 0.0
            else:
                err = 0.0
        return err

    def _get_p_convergence_rate(self):
        # get largest coefficients per total degree
        p_max = np.max(np.sum(self.idx_act_gpc, axis=1))
        coeff_ls = []
        for p in range(1, p_max+1):
            false_true = np.where(np.sum(self.idx_act_gpc, axis=1) == p)
            coeff_ls.append(np.max(np.abs(self.pce_coeff_sur)[false_true]))

        # least squares to estimate convergence rate
        b = np.log(np.abs(coeff_ls))
        A = np.vstack([np.ones(len(coeff_ls)), -np.arange(1, p_max+1)]).T
        C = scl.lstsq(A, b)
        c = np.exp(C[0][0])
        m = C[0][1]

        return c, m

    def _approximate(self):
        """
        Get approximation on leaf
        """
        # leja stabilization constant
        beta = 1

        # get total degree index set to total degree p_max
        if self.basis == 'TD':
            idx_set = td_multiindex_set(self.N, self.p_max)
        else:
            raise RuntimeError(
                'Basis for (python) Leja nodes must be \"TD\"')

        # maximum index per dimension
        max_idx_per_dim = np.max(idx_set, axis=0)

        # prepare univariate knots and polynomials per dimension
        knots_per_dim = {}
        self.polys_per_dim = {}

        # delete old nodes if they should not be reused
        if not self.reuse_Leja_nodes_href:
            self.knots_per_dim = {}
            for n in range(self.N):
                self.knots_per_dim[n] = np.array([]).reshape(-1, 1)

        # determine necessary number of leja points per dimension
        nr_knots_per_dim = {}
        for n in range(self.N):
            if self.dim_refined is not None and self.sort_Leja_nodes and self.dim_refined == n:
                needed_knots = beta*self.knots_per_dim[n].shape[0]
                nr_knots_per_dim[n] = np.max(
                    [max_idx_per_dim[n], needed_knots])
            else:
                nr_knots_per_dim[n] = max_idx_per_dim[n]

        # get univariate Leja nodes
        for n in range(self.N):
            # univariate knots per dimension
            if self.knots_per_dim[n].shape[0] < nr_knots_per_dim[n] + 1:
                # new nodes necessary
                kk, _ = seq_lj_1d(
                    nr_knots_per_dim[n], self.dist[n], self.knots_per_dim[n], self.discontinuous)
                knots_per_dim[n] = kk
            else:
                # use old nodes
                knots_per_dim[n] = self.knots_per_dim[n]

        # save knots at tree node
        self.knots_per_dim = knots_per_dim

        # re-order nodes if dimension has been h-refined
        if self.dim_refined is not None and self.sort_Leja_nodes:
            self.knots_per_dim[self.dim_refined] = lejaorder(
                [self.knots_per_dim[self.dim_refined].copy()])[0]

        # construct univariate hierarchical polynomials per dimension
        for n in range(self.N):
            # no. of knots = no. of polynomials = P_n+1
            P = len(knots_per_dim[n])
            self.polys_per_dim[n] = [Hierarchical1d(
                knots_per_dim[n][:p+1]) for p in range(P)]

        # get knots for total degree basis
        knots = np.array([[self.knots_per_dim[n][i] for
                          n, i in zip(range(self.N), idx_set[j])] for
                          j in range(len(idx_set))])

        # save number of initial function calls
        self.fcalls_init = self.data.shape[0]

        # check if p- or h-refinement
        # if p-ref, exploit nestedness of Leja nodes and start computing
        # the surpluses with the new nodes
        try:
            # p-refinement!
            self.idx_act
            fevals = self.data[:, -1].tolist()
            idx_shift = len(fevals)
        except Exception:
            idx_shift = 0
            self.idx_act = []
            self.hs_act = []
            self.hs2_act = []
            fevals = []
        self.fcalls_local = 0
        self.reused_mask = np.zeros(knots.shape[0], dtype=bool)
        for i in range(idx_shift, knots.shape[0]):
            # check if knot is already calculated
            if idx_shift == 0:  # -> h-refinement
                if self.data.shape[0] > 0 and self.reuse_Leja_nodes_href:
                    dist = np.linalg.norm(
                        self.data[:, :-1] - knots[i, :], 1, axis=1)
                    mask = dist < np.finfo(float).eps
                    if np.sum(mask) > 1:
                        raise RuntimeError('Found two identical old nodes!')
                    else:
                        idx = mask.nonzero()[0]

                else:
                    idx = np.array([])
                if idx.shape[0] == 0:
                    feval = self.f(knots[i].reshape(1, -1)).flatten()[0]
                    Tree.fcalls += 1
                    self.fcalls_local += 1
                else:
                    feval = self.data[idx, -1][0]
                    self.reused_mask[i] = True
            else:
                feval = self.f(knots[i].reshape(1, -1)).flatten()[0]
                Tree.fcalls += 1
                self.fcalls_local += 1

            if len(self.hs_act) == 0:
                self.hs_act.append(feval)
                self.hs2_act.append(feval**2)
            else:
                ieval = interpolate_single(
                    self.idx_act, self.hs_act, self.polys_per_dim, knots[i])
                self.hs_act.append(feval-ieval)

            self.idx_act.append(idx_set[i].tolist())
            fevals.append(feval)

        # save data: [parameter 1, parameter 2, ..., function value]
        self.data = np.hstack([knots, np.array(fevals).reshape(-1, 1)])

        if idx_shift == 0:  # -> h-refinement
            #                 local evaluations -  |      needed evaluations           |
            Tree.fcalls_lost += self.fcalls_local - \
                (self.data.shape[0] - self.fcalls_init)
            Tree.fcalls_possible_saving += self.fcalls_init
        # compute pce basis
        self._construct_local_pce_model()

        # leaf approximated
        self.approximated = True
        self.dim_refined = None

    def get_moments(self):
        """
        Return mean and variance.
        """
        leaves = self.get_leaves()
        # moments from PCE model
        # get global mean
        mean_PCE = np.sum([[leaf.local_mean_PCE*leaf.Jk] for leaf in leaves])
        variance_PCE = np.sum(
            [[(leaf.local_variance_PCE+(leaf.local_mean_PCE-mean_PCE)**2)*leaf.Jk] for leaf in leaves])
        return [mean_PCE, variance_PCE]

    def _search_and_refine(self, theta1, theta2, thetap):
        # continue search of child exists
        if self.left:
            self.left._search_and_refine(theta1, theta2, thetap)

        if self.right:
            self.right._search_and_refine(theta1, theta2, thetap)

        # if node is leaf, check if refinement is necesarry
        if not self.left and not self.right and self.approximated and self.refinable and self.mark:
            self._refine(theta2, thetap)

    def _search_and_approximate(self):

        # continue search if child exists
        if self.left:
            self.left._search_and_approximate()

        if self.right:
            self.right._search_and_approximate()

        # if node is leaf, start gPC approximation
        if not self.left and not self.right and not self.approximated:
            self._approximate()

    def get_no_of_cells(self):
        """
        Return number of elements. 
        First number gives total number of elements, 
        second number given total number of refineable elements.
        """
        if self.left:
            no_left, no_refineable_left = self.left.get_no_of_cells()

        if self.right:
            no_right, no_refineable_right = self.right.get_no_of_cells()

        # if node is leaf, start gPC approximation
        if not self.left and not self.right:
            if self.refinable:
                return 1, 1
            else:
                return 1, 0
        else:
            return no_left + no_right, no_refineable_left + no_refineable_right

    def get_data(self):
        """
        Return used nodes and function evaluations.
        """
        data = np.empty([0, self.N+1])
        if self.left:
            data = np.append(data, self.left.get_data(), axis=0)

        if self.right:
            data = np.append(data, self.right.get_data(), axis=0)

        # if node is leaf, start gPC approximation
        if not self.left and not self.right:
            local_data = np.copy(self.data)
            for i in range(self.N):
                local_data = local_data
            return local_data
        else:
            return data

    def _construct_local_pce_model(self):
        # get jpdf from openturns
        self.dist_ot = get_dist_ot(self.dist)
        # get multivariate basis
        self.mv_basis = get_mv_basis_ot(self.dist_ot)
        # get enumeration function
        self.enum_func = self.mv_basis.getEnumerateFunction()
        # create PCE basis
        idx_system = self.idx_act
        self.idx_act_gpc = idx_system
        idx_system_single = transform_multi_index_set(
            idx_system, self.enum_func)
        self.system_basis = self.mv_basis.getSubBasis(idx_system_single)
        # get corresponding evaluations
        fevals_system = self.data[:, -1]
        # multi-dimensional knots
        knots_md = self.data[:, :-1].copy()
        # tranform knots to local boundaries
        knots_md = map_data(self.dist, knots_md)
        # design matrix
        self.D = get_design_matrix(self.system_basis, knots_md)
        # solve system of equaations
        Q, R = scl.qr(self.D, mode='economic')
        c = Q.T.dot(fevals_system)
        self.pce_coeff_sur = scl.solve_triangular(R, c)
        [mu, sigma2] = compute_moments(self.pce_coeff_sur)
        self.local_variance_PCE = sigma2
        self.local_mean_PCE = mu

    def evaluate(self, X):
        """
        Evaluate surrogate model.
        """
        X = np.array(X)
        # empty vector for interpolation values
        Z = np.empty((X.shape[0], 1))
        Z.fill(np.nan)
        # get leaves
        leaves = self.get_leaves()
        for leaf in leaves:
            bounds = leaf.bounds
            idx = []
            # get data that suits to the bounds of the current leaf
            for n in range(self.N):
                b = bounds[n]
                idx.append(
                    set(np.where(np.logical_and(X[:, n] >= b[0], X[:, n] <= b[1]))[0].tolist()))

            idx_inter = idx[0]
            for n in range(self.N-1):
                idx_inter = idx_inter.intersection(idx[n+1])
            idx_inter = list(idx_inter)

            Xeval = X[idx_inter]
            # evaluate leaf
            ievals = leaf._evaluate(Xeval)
            Z[idx_inter] = ievals.reshape(-1, 1)

        return Z

    def run(self, theta1=0.00001, theta2=0.1, thetap=0.5, no_refinements=None, verbose=False):
        """
        Run the multi-element algorithm with hp-refinement.
        """
        Tree.verbose = verbose

        # initial start
        if self.id == 1:
            if not self.approximated:
                self._search_and_approximate()
            else:
                if self.left is None and self.right is None:
                    self.mark = False
                    self.refinable = True
                    self._mark_leaves(theta1, [self])
                else:
                    leaves = self.get_leaves()
                    for leaf in leaves:
                        # get local error indicators
                        leaf.local_error_ind = leaf._get_local_error_indicator()
                        # unmark leaves
                        leaf.mark = False
                        leaf.refinable = True
                    # mark leaves
                    self._mark_leaves(theta1, leaves)

        while True:
            no_cells, no_refinable_cells = self.get_no_of_cells()
            if no_refinable_cells == 0:
                break
            if no_refinements is not None:
                if no_refinements <= self.k:
                    break
            leaves = self.get_leaves()
            for leaf in leaves:
                # get local error indicators
                leaf.local_error_ind = leaf._get_local_error_indicator()
                # unmark leaves
                leaf.mark = False
                leaf.refinable = True
            # mark leaves
            self._mark_leaves(theta1, leaves)
            # refine leaves
            self._search_and_refine(theta1, theta2, thetap)
            self._search_and_approximate()
            self.k += 1

    def get_leaves(self, leaves=None):
        """
        Return all leaves.
        """
        if self.id == 1:
            leaves = []
        if self.left:
            self.left.get_leaves(leaves)
        if self.right:
            self.right.get_leaves(leaves)
        # if node is leaf, start gPC approximation
        if not self.left and not self.right:
            leaves.append(self)
            if self.id == 1:
                return leaves
            return
        else:
            return leaves

    def get_total_fcalls(self, fcalls=0):
        """
        Return total number of function evaluations.
        """
        fcalls += self.fcalls_local
        if self.left:
            fcalls = self.left.get_total_fcalls(fcalls)
        if self.right:
            fcalls = self.right.get_total_fcalls(fcalls)

        return fcalls

    def get_error_per_leaf(self, cv_in, cv_out, verbose=True):
        """
        Returns all leaves with estimated approximation error.
        The errors per leaf can be accessed via \n
        leaf.cv_error_max \n
        leaf.cv_error_mean \n
        leaf.cv_error_rms
        """
        leaves = self.get_leaves()
        for leaf in leaves:
            local_cv_in, local_cv_out = self._sort_data_to_bounds(
                cv_in, leaf.bounds, data_out=cv_out)
            local_cv_sur = []
            if local_cv_in.shape[0] > 0:
                local_cv_sur = leaf._evaluate(local_cv_in)
                errors = np.abs(local_cv_sur-local_cv_out)
                leaf.cv_error_max = np.max(errors)
                leaf.cv_error_mean = np.mean(errors)
                leaf.cv_error_rms = np.sqrt(
                    np.sum(errors**2)/local_cv_in.shape[0])
                if verbose:
                    print('bounds: ' + str(leaf.bounds) +
                          ' , Jk: ' + str(leaf.Jk))
                    print('leaf id: ' + str(leaf.id))
                    # print('max cv error: ', leaf.cv_error_max)
                    # print('mean cv error: ', leaf.cv_error_mean)
                    print('RMS cv error: ', leaf.cv_error_rms)
                    print('')
            else:
                leaf.cv_error_max = 0
                leaf.cv_error_mean = 0
                leaf.cv_error_rms = 0
                if verbose:
                    print('no cv samples on leaf')
                    print('')
        return leaves

    def plot_mesh(self, fig=None):
        """
        Plot the parameter domain decomposition. Only for N=1 or N=2.
        """
        if self.id == 1:
            fig = plt.figure()
            # plt.gca().set_aspect('equal', adjustable='box')
        if self.N == 2:
            # continue search if child exists
            if self.left:
                self.left.plot_mesh(fig)

            if self.right:
                self.right.plot_mesh(fig)

            # if node is leaf, start gPC approximation
            if not self.left and not self.right:
                plt.plot(self.bounds[0], [self.bounds[1][0],
                         self.bounds[1][0]], color='b')  # X
                plt.plot(self.bounds[0], [self.bounds[1][1],
                         self.bounds[1][1]], color='b')  # X
                plt.plot([self.bounds[0][0], self.bounds[0][0]],
                         self.bounds[1], color='b')  # Y
                plt.plot([self.bounds[0][1], self.bounds[0][1]],
                         self.bounds[1], color='b')  # Y

        elif self.N == 1:
            # plot 1D data
            if self.bounds[0] == [-1, 1]:
                data_sorted = self.data[self.data[:, 0].argsort()]
                plt.plot(data_sorted[:, 0], data_sorted[:, 1], '.', color='g')

            # continue search if child exists
            if self.left:
                self.left.plot_mesh(fig)

            if self.right:
                self.right.plot_mesh(fig)

            # if node is leaf, start gPC approximation
            if not self.left and not self.right:
                y_max = np.max(self.data[:, 1])
                plt.plot([self.bounds[0][0], self.bounds[0][0]],
                         [-0.1, 0.1], color='b')  # X
                plt.plot([self.bounds[0][1], self.bounds[0][1]],
                         [-0.1, 0.1], color='b')  # X
                # plt.plot(map_data(self.data,[-1,1],self.bounds[0],0)[:,0],self.data[:,1],'.', color='g') # X
                plt.plot(self.data[:, 0], self.data[:, 1], '.', color='g')  # X
                # plt.plot([self.bounds[0][0],self.bounds[0][0]],[0,y_max], color='b') # X
                # plt.plot([self.bounds[0][1],self.bounds[0][1]],[0,y_max], color='b') # X

        else:
            print('only implemented for N=1 or N=2')
        return fig

    def plot_pmax_on_mesh(self, colormap='cubehelix', save_to_file=False):
        """ 
        Plot polynomial degree on mesh. Only for N=1 and N=2.
        """
        if self.N == 2:
            if self.id == 1:
                leaves = self.get_leaves()
                p_maxz = []
                for leaf in leaves:
                    p_maxz.append(leaf.p_max)
                plt.figure()
                p_min = np.min(p_maxz)
                p_max = np.max(p_maxz)

                cnorm = mcol.Normalize(vmin=p_min, vmax=p_max)
                cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
                cpick.set_array([])
                plt.colorbar(cpick, label='pmax')
                # plt.gca().set_aspect('equal', adjustable='box')

            for leaf in leaves:
                cnorm = mcol.Normalize(vmin=p_min, vmax=p_max)
                cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
                cpick.set_array([])
                plt.gca().fill_between(
                    leaf.bounds[0], leaf.bounds[1][0], leaf.bounds[1][1], color=cpick.to_rgba(leaf.p_max))
                plt.plot(leaf.bounds[0], [leaf.bounds[1][0],
                         leaf.bounds[1][0]], color='gray')  # X
                plt.plot(leaf.bounds[0], [leaf.bounds[1][1],
                         leaf.bounds[1][1]], color='gray')  # X
                plt.plot([leaf.bounds[0][0], leaf.bounds[0][0]],
                         leaf.bounds[1], color='gray')  # Y
                plt.plot([leaf.bounds[0][1], leaf.bounds[0][1]],
                         leaf.bounds[1], color='gray')  # Y

        elif self.N == 1:
            if self.id == 1:
                leaves = self.get_leaves()
                p_maxz = [leaf.p_max for leaf in leaves]
                p_min = np.min(p_maxz)
                p_max = np.max(p_maxz)

                cnorm = mcol.Normalize(vmin=p_min, vmax=p_max)
                cpick = cm.ScalarMappable(norm=cnorm, cmap='viridis')
                cpick.set_array([])

                # plt.gca().set_aspect('equal', adjustable='box')

                fig, ax = plt.subplots()
                ax.set_xlabel("x")
                ax.set_ylabel("pmax")
                ax2 = ax.twinx()
                ax2.set_ylabel("g(x)")

                plt.figure()
                plt.colorbar(cpick, label='pmax')
                XXfunc = np.linspace(
                    self.bounds[0][0], self.bounds[0][1], 100).reshape(-1, 1)
                YYfunc = self.evaluate(XXfunc)
                YYref = [self.f(XXfunc_eval) for XXfunc_eval in XXfunc]
                ax2.plot(XXfunc, YYfunc, color='gray')
                ax2.plot(XXfunc, YYref, color='blue')

                ii = 1
                for leaf in leaves:
                    bounds = leaf.bounds
                    XX = [bounds[0][0], bounds[0][1]]
                    YY = [leaf.p_max, leaf.p_max]
                    # color=cpick.to_rgba(leaf.p_max))
                    ax.plot(XX, YY, 'o-', color='black')
                    MM = np.array([XX, YY]).T
                    if save_to_file:
                        np.savetxt('bounds_pmax_' + str(ii) + '.txt', MM)
                    ii += 1

        else:
            print('only implemented for N=1 and N=2')

    def plot_local_error_indicator_on_mesh(self, colormap='cubehelix'):
        """
        Plot local error indicator on mesh. Only for N=2.
        """
        if self.N == 2:
            if self.id == 1:
                leaves = self.get_leaves()
                local_error_ind = []
                for leaf in leaves:
                    local_error_ind.append(leaf._get_local_error_indicator())
                plt.figure()
                indicator_min = np.min(local_error_ind)
                indicator_max = np.max(local_error_ind)

                cnorm = mcol.Normalize(vmin=indicator_min, vmax=indicator_max)
                cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
                cpick.set_array([])
                plt.colorbar(cpick, label='eta_k')

            for i, leaf in enumerate(leaves):
                cnorm = mcol.Normalize(vmin=indicator_min, vmax=indicator_max)
                cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
                cpick.set_array([])
                plt.gca().fill_between(
                    leaf.bounds[0], leaf.bounds[1][0], leaf.bounds[1][1], color=cpick.to_rgba(local_error_ind[i]))
                plt.plot(leaf.bounds[0], [leaf.bounds[1][0],
                         leaf.bounds[1][0]], color='gray')  # X
                plt.plot(leaf.bounds[0], [leaf.bounds[1][1],
                         leaf.bounds[1][1]], color='gray')  # X
                plt.plot([leaf.bounds[0][0], leaf.bounds[0][0]],
                         leaf.bounds[1], color='gray')  # Y
                plt.plot([leaf.bounds[0][1], leaf.bounds[0][1]],
                         leaf.bounds[1], color='gray')  # Y

        else:
            print('only implemented for N=2')

    def plot_pc_coefficients(self):
        """
        Plot polynomial chaos coefficients.
        """
        leaves = self.get_leaves()

        for leaf in leaves:
            # get convergence rate
            c, m = leaf._get_p_convergence_rate()
            # start figure
            plt.figure()
            plt.xlabel('|p|')
            plt.ylabel('|s_p|')
            plt.title('#' + str(leaf.id) + ' ' + str(leaf.bounds) + ' pmax = ' +
                      str(leaf.p_max) + ' \n theta: {:.3e}'.format(np.exp(-m)))
            # get colormap and scale data to 0-1
            cm_act = cm.jet
            if leaf.N >= 2:
                cm_plot = [cm_act(1./(leaf.N-1)*(i-1))
                           for i in list(range(1, leaf.N+1))]
            else:
                cm_plot = [cm_act(0.5)]
            for n in range(leaf.N):
                idx_leaf = leaf.idx_act.copy()
                coeff_leaf = leaf.pce_coeff_sur.tolist().copy()
                p_max_uni = np.max(np.array(leaf.idx_act)[:, 0])
                # plot univariate coefficients first
                for p in range(1, p_max_uni+1):
                    log1 = np.sum(idx_leaf, axis=1) == p
                    log2 = np.array(idx_leaf)[:, n] == p
                    idx_tmp = np.where(np.logical_and(log1, log2))[0]
                    if idx_tmp.shape[0] != 0:
                        idx = np.where(np.logical_and(log1, log2))[0][0]
                        coeff_plot = np.abs(coeff_leaf.pop(idx))
                        p_plot = np.sum(idx_leaf.pop(idx))
                        plt.semilogy(p_plot, coeff_plot, 'd',
                                     color=cm_plot[n], markersize=10)

                # plot remaining coefficients
                for i, value in enumerate(idx_leaf):
                    p_plot = np.sum(value)
                    coeff_plot = np.abs(coeff_leaf[i])
                    plt.semilogy(p_plot, coeff_plot, '.', color='black')
                # plot convergence rate
                p_max = np.max(np.sum(leaf.idx_act, axis=1))
                plt.semilogy(np.arange(1, p_max+1), c *
                             np.exp(-np.arange(1, p_max+1)*m), '--', color='black')

    def plot_error_on_mesh(self, cv_in, cv_out, error_norm='rms', colormap='cubehelix', verbose=True):
        """
        Plot error on mesh. Implemented for 1D and 2D.
        """
        if self.id == 1:
            leaves = self.get_error_per_leaf(cv_in, cv_out, verbose=verbose)
            cv_error_max = []
            cv_error_mean = []
            cv_error_rms = []
            for leaf in leaves:
                cv_error_max.append(leaf.cv_error_max)
                cv_error_mean.append(leaf.cv_error_mean)
                cv_error_rms.append(leaf.cv_error_rms)
            plt.figure()
            if error_norm == 'max':
                min_error = np.min(cv_error_max)
                max_error = np.max(cv_error_max)
            elif error_norm == 'mean':
                min_error = np.min(cv_error_mean)
                max_error = np.max(cv_error_mean)
            elif error_norm == 'rms':
                min_error = np.min(cv_error_rms)
                max_error = np.max(cv_error_rms)
            if min_error == 0:
                Tree.min_error = np.finfo(float).eps
            else:
                Tree.min_error = min_error
            Tree.max_error = max_error
            cnorm = mcol.LogNorm(vmin=Tree.min_error, vmax=Tree.max_error)
            cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
            cpick.set_array([])
            plt.colorbar(cpick, label=error_norm + ' error')
            # plt.gca().set_aspect('equal', adjustable='box')
        if self.N == 2:
            # continue search if child exists
            if self.left:
                self.left.plot_error_on_mesh(error_norm, colormap)

            if self.right:
                self.right.plot_error_on_mesh(error_norm, colormap)

            # if node is leaf, start gPC approximation
            if not self.left and not self.right:
                if error_norm == 'max':
                    local_error = self.cv_error_max
                elif error_norm == 'mean':
                    local_error = self.cv_error_mean
                elif error_norm == 'rms':
                    local_error = self.cv_error_rms

                cnorm = mcol.LogNorm(vmin=Tree.min_error, vmax=Tree.max_error)
                cpick = cm.ScalarMappable(norm=cnorm, cmap=colormap)
                cpick.set_array([])
                plt.gca().fill_between(
                    self.bounds[0], self.bounds[1][0], self.bounds[1][1], color=cpick.to_rgba(local_error))

        else:
            print('only implemented for N=2')

    def _sort_data_to_bounds(self, data_in, bounds, data_out=None):
        for (i, bound) in enumerate(bounds):
            lower_bound = bound[0]
            upper_bound = bound[1]
            idx = np.where(data_in[:, i] > lower_bound)[0]
            if data_out is not None:
                data_out = data_out[idx]
            data_in = data_in[idx, :]
            idx = np.where(data_in[:, i] < upper_bound)[0]
            if data_out is not None:
                data_out = data_out[idx]
            data_in = data_in[idx, :]
        if data_out is None:
            return data_in
        else:
            return data_in, data_out

    def _evaluate(self, X):
        return interpolate_multiple(self.idx_act, self.hs_act, self.polys_per_dim, X)

    def get_leaf(self, idx):
        """
        Return the leaf with a certain id
        """
        leaves = self.get_leaves()
        for leaf in leaves:
            if leaf.id == idx:
                return leaf

        return None

    def _write_parameters(self):

        Tree.nodes = Tree.parameter['nodes']
        Tree.href_split_marker = Tree.parameter['href_split_marker']
        Tree.marking_strategy = Tree.parameter['marking_strategy']
        Tree.basis = Tree.parameter['basis']

        Tree.reuse_Leja_nodes_href = Tree.parameter_leja['reuse_Leja_nodes_href']
        Tree.sort_Leja_nodes = Tree.parameter_leja['sort_Leja_nodes']

        Tree.use_exact_error = Tree.parameter['use_exact_error']

    def get_savings(self, fcalls_init=0, fcalls_local=0, fcalls_total=0):
        """
        Return savings through h-refinement.
        """
        fcalls_init += self.fcalls_init
        fcalls_local += self.fcalls_local
        fcalls_total += self.data.shape[0]
        if self.left:
            fcalls_init, fcalls_local, fcalls_total = self.left.get_savings(
                fcalls_init, fcalls_local, fcalls_total)

        if self.right:
            fcalls_init, fcalls_local, fcalls_total = self.right.get_savings(
                fcalls_init, fcalls_local, fcalls_total)

        if self.id == 1:
            savings_abs = fcalls_total - fcalls_local
            if fcalls_init != 0:
                savings_rel = (fcalls_total - fcalls_local)/fcalls_init
            else:
                savings_rel = np.nan

            print('saved {} function calls, which is {:5.2f}% of possible savings'.format(
                savings_abs, savings_rel*100))
        return fcalls_init, fcalls_local, fcalls_total
