
public class LDA {

    public static void main(String args[]) throws Exception {

        /*
        for (String s: args) {
            System.out.println(s);
        }
        */

        String word_num_file = "/Users/dcard/Projects/CMU/ARK/guac/datasets/mfc_v2/lda/word_num.json";
        String word_doc_file = "/Users/dcard/Projects/CMU/ARK/guac/datasets/mfc_v2/lda/word_doc.json";
        String vocab_file = "/Users/dcard/Projects/CMU/ARK/guac/datasets/mfc_v2/lda/vocab.json";

        Sampler sampler = new Sampler(1, 1, 10);
        sampler.loadData(word_num_file, word_doc_file, vocab_file);
        sampler.run(10, 1, 1, 200, 40, 1);


        /*
        String vocab_file = "/Users/dcard/Projects/CMU/ARK/guac/datasets/mfc_full/lda/vocab.json";
        JSONObject vocab = (JSONObject) parser.parse(new FileReader(vocab_file));
        Iterator<Integer> iterator = vocab_assignments.iterator();
        for (int i=0; i < 10; i++) {
            int  index = iterator.next();
            System.out.println(index + " " + vocab.get(index));
        }
        */


    }

}
