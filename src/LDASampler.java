import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import java.io.FileReader;
import java.util.concurrent.ThreadLocalRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Random;
import java.nio.file.Path;

public class LDASampler {
    int n_topics;
    float alpha;
    float beta;

    int vocab_size;
    int n_words;
    int n_docs;

    int doc_assignments[];
    int vocab_assignments[];
    String vocab[];

    int topic_assignments[];
    int topic_counts[];
    int vocab_topics[][];
    int doc_topics[][];

    int t_topic_counts[];
    int t_vocab_topics[][];
    int t_doc_topics[][];


    public LDASampler(Path word_file, Path doc_file, Path vocab_file) throws Exception {
        JSONParser parser = new JSONParser();
        JSONArray words = (JSONArray) parser.parse(new FileReader(word_file.toString()));
        JSONArray docs = (JSONArray) parser.parse(new FileReader(doc_file.toString()));
        JSONArray vocab_json = (JSONArray) parser.parse(new FileReader(vocab_file.toString()));

        n_words = words.size();

        vocab_size = 0;
        vocab_assignments = new int[n_words];

        for (int i = 0; i < n_words; i++) {
            vocab_assignments[i] = ((Long) words.get(i)).intValue();
            if (vocab_assignments[i] > vocab_size) {
                vocab_size = vocab_assignments[i];
            }
        }
        vocab_size += 1;

        n_docs = 0;
        doc_assignments = new int[n_words];
        for (int i = 0; i < n_words; i++) {
            doc_assignments[i] = ((Long) docs.get(i)).intValue();
            if (doc_assignments[i] > n_docs) {
                n_docs = doc_assignments[i];
            }
        }
        n_docs += 1;

        vocab = new String[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            vocab[i] = (String) vocab_json.get(i);
        }

    }

    public String[] get_vocab() {
        return vocab;
    }

    public int[][] run(int n_topics, float alpha, float beta, int n_iter, int burn_in, int subsampling) {
        this.n_topics = n_topics;
        this.alpha = alpha;
        this.beta = beta;

        // initialize arrays
        System.out.println("Initializing arrays");
        topic_assignments = new int[n_words];
        topic_counts = new int[n_topics];
        vocab_topics = new int[vocab_size][n_topics];
        doc_topics = new int[n_docs][n_topics];

        t_topic_counts = new int[n_topics];
        t_vocab_topics = new int[vocab_size][n_topics];
        t_doc_topics = new int[n_docs][n_topics];

        // do random initalization
        for (int j=0; j < n_words; j++) {
            int d_j = doc_assignments[j];
            int v_j = vocab_assignments[j];
            int k = ThreadLocalRandom.current().nextInt(0, n_topics);

            topic_assignments[j] = k;
            topic_counts[k] += 1;
            vocab_topics[v_j][k] += 1;
            doc_topics[d_j][k] += 1;

        }

        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n_words; i++) {
            order.add(i);
        }
        Collections.shuffle(order);
        Random rand = new Random();

        System.out.println("Doing burn-in");
        for (int i=0; i < n_iter; i++) {
            for (int q=0; q < n_words; q++) {
                //double p[] = new double[n_topics];
                double p[] = new double[n_topics];
                int j = order.get(q);
                int d_j = doc_assignments[j];
                int z_j = topic_assignments[j];
                int v_j = vocab_assignments[j];

                assert topic_counts[z_j] > 0;
                assert vocab_topics[v_j][z_j] > 0;
                assert doc_topics[d_j][z_j] > 0;

                // remove the counts for this word
                topic_counts[z_j] -= 1;
                vocab_topics[v_j][z_j] -= 1;
                doc_topics[d_j][z_j] -= 1;

                // compute probabilities
                double p_sum = 0;
                for (int k = 0; k < n_topics; k++) {
                    p[k] = (alpha + doc_topics[d_j][k]) * (beta + vocab_topics[v_j][k]) / (float) (vocab_size * beta + topic_counts[k]);
                    assert p[k] > 0;
                    p_sum += p[k];
                }

                // sample a topic
                double f = rand.nextDouble() * p_sum;
                int k = 0;
                double temp = p[k];
                while (f > temp) {
                    k += 1;
                    temp += p[k];
                }

                topic_counts[k] += 1;
                vocab_topics[v_j][k] += 1;
                doc_topics[d_j][k] += 1;
                topic_assignments[j] = k;

            }


            if (i % subsampling == 0) {
                if (i > burn_in)
                    System.out.print(".");
                else
                    System.out.print("-");
                for (int k = 0; k < n_topics; k++) {
                    t_topic_counts[k] += topic_counts[k];
                    for (int d = 0; d < n_docs; d++)
                        t_doc_topics[d][k] += doc_topics[d][k];
                    for (int v = 0; v < vocab_size; v++)
                        t_vocab_topics[v][k] += vocab_topics[v][k];
                }
            }
        }

        // return final word-topic matrices

        /*
        for (int k=0; k < n_topics; k++) {
            System.out.println(k);
            List<Integer> list = new ArrayList<>();
            for (int v = 0; v < vocab_size; v++)
                list.add(t_vocab_topics[v][k]);

            Collections.sort(list);
            Collections.reverse(list);
            int n_to_print = 30;
            int threshold = list.get(n_to_print);
            for (int v = 0; v < vocab_size; v++) {
                if (t_vocab_topics[v][k] >= threshold)
                    System.out.println(vocab[v] + ": " + t_vocab_topics[v][k]);
            }
        }
        */

        return t_vocab_topics;
    }

}
