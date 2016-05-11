import org.ejml.data.DenseMatrix64F;

/*
Copied from the Gaussian_LDA GitHub repo: https://github.com/rajarshd/Gaussian_LDA.git
which accompanies the paper "Gaussian LDA for Topic Models with Word Embeddings"
Citation:
@InProceedings{das-zaheer-dyer:2015,
  author    = {Das, Rajarshi  and  Zaheer, Manzil  and  Dyer, Chris},
  title     = {Gaussian LDA for Topic Models with Word Embeddings},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  publisher = {Association for Computational Linguistics},
  url       = {http://www.aclweb.org/anthology/P15-1077}
}
*/

public class NormalInverseWishart {

    /**
     * Hyperparam mean vector.
     */
    public  DenseMatrix64F mu_0 ;

    /**
     * initial degrees of freedom
     */
    public  double nu_0;

    /**
     * Hyperparam covariance matrix
     */
    public  DenseMatrix64F sigma_0 ;

    /**
     * mean fraction
     */
    public  double k_0;



    public  DenseMatrix64F getMu_0() {
        return mu_0;
    }

    public  void setMu_0(DenseMatrix64F mu_0) {
        this.mu_0 = mu_0;
    }

    public  double getNu_0() {
        return nu_0;
    }

    public  void setNu_0(double nu_0) {
        this.nu_0 = nu_0;
    }

    public  DenseMatrix64F getSigma_0() {
        return sigma_0;
    }

    public  void setSigma_0(DenseMatrix64F sigma_0) {
        this.sigma_0 = sigma_0;
    }

    public  double getK_0() {
        return k_0;
    }

    public  void setK_0(double k_0) {
        this.k_0 = k_0;
    }



}
