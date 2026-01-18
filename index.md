<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cascaded Transfer: Learning Many Tasks under Budget Constraints | submitted to ICML 2026</title>
    
    <!-- Polices de caractères professionnelles depuis Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
    
    <!-- Lien vers votre feuille de style CSS -->
    <link rel="stylesheet" href="style.css">
</head>
<body>

    <div class="container">

        <header>
                    <div class="project-header">
                <div class="project-title-container">
                    <span class="project-name">CASCADER</span>
                    <span class="project-acronym">Cascaded Adaptive Source-to-Client Transfer via Dynamic Embedding and  Retraining</span>
                </div>
                <img src="images/logo.png" alt="CASCADER Logo" class="project-logo">
            </div>
            <h1 class="paper-title">[Titre Complet de Votre Papier sur le Transfert Learning en Cascade]</h1>
            <p class="conference">Submitted to ICML 2026</p>
            
            <div class="authors">
                <!-- 
                <a href="[Lien vers votre page personnelle]">Votre Nom</a><sup>1</sup>,
                <a href="[Lien vers la page du co-auteur]">Nom du Co-auteur</a><sup>2</sup>,
                <a href="[Lien vers la page du co-auteur]">Autre Co-auteur</a><sup>1,2</sup> 
                -->
                CASCADER: Cascaded Adaptive Source-to-Client Transfer via Dynamic Embedding and Retraining
            </div>
            
            <!--
            <div class="affiliations">
                <sup>1</sup>[Nom de votre Université/Laboratoire] &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>[Nom de l'autre institution]
            </div>
            -->

            <nav class="links">
                <a href="[Lien vers votre PDF, ex: ./paper.pdf]" class="button">📄 Papier</a>
                <a href="[Lien vers votre dépôt GitHub]" class="button">💻 Code</a>
                <a href="[Lien vers une vidéo de présentation, si applicable]" class="button">🎥 Vidéo</a>
                <a href="#citation" class="button">✏️ Citer</a>
            </nav>
        </header>

        <main>
            <section class="teaser">
                <img src="images/cascade.png" alt="The Cascaded Transfer Learning (CTL) paradigm. (a) A collection of related tasks visualized over a landscape in which proximity represents task relatedness. A seed task (red) is selected among all tasks. (b) Cascaded transfer starting from the seed and refining each task in turn along a tree. Darker red color shades indicate nodes refined earlier in the process.">
                <figcaption>
                    <b>Figure 1 :</b> [Brève description de l'image principale. Par exemple : Notre approche de transfert learning en cascade, où les connaissances sont transférées séquentiellement à travers une série de tâches pour améliorer la performance finale.]
                </figcaption>
            </section>

            <!-- Section Abstract -->
            <section id="abstract">
                <h2>Abstract</h2>
                <p>
Many-Task Learning refers to the setting where a large number of related tasks need to be learned, the exact relationships between tasks are not known, and budget constraints are in place. We introduce the Cascaded Transfer Learning (CTL), a novel many-task transfer learning paradigm where information (e.g. model parameters) cascades hierarchically though tasks that are learned by individual models of the same class, while respecting given budget constraints. The cascade hierarchy is incrementally built by finding ordered task triplets (source, intermediate, target), where the intermediate task is chosen to facilitate the source-target transfer. We give the conditions under which a transfer passing through aintermediate task improves over a direct source-target transfer. Based on these insights, we design a cascaded transfer mechanism that is deployed over a tree structure connecting the tasks, and allocates the available training budget along its branches. Experiments on synthetic and real many-task settings show that the resulting method enables more accurate and cost-effective adaptation across large task collections compared to alternative approaches.
                </p>
            </section>

            <!-- Section Méthode -->
            <section id="method">
                <h2>Notre Méthode</h2>
                <p>
                    [Décrivez ici de manière concise votre méthode. Vous pouvez expliquer le concept de "cascade". Par exemple : Notre méthode s'articule autour d'une architecture de transfert en cascade. Contrairement aux approches traditionnelles qui transfèrent les connaissances d'une seule source à une seule cible, notre modèle crée une chaîne d'apprentissage où chaque modèle intermédiaire sert de fondation pour le suivant, spécialisant progressivement les features pour des tâches de plus en plus complexes.]
                </p>
                <img src="images/method_diagram.svg" alt="Diagramme de l'architecture de la méthode">
                <figcaption>
                    <b>Figure 2 :</b> [Description du diagramme de votre méthode.]
                </figcaption>
            </section>

            <!-- Section Résultats -->
            <section id="results">
                <h2>Résultats Expérimentaux</h2>
                <p>
                    [Présentez vos résultats principaux. Par exemple : Nous avons évalué notre approche sur plusieurs benchmarks de vision par ordinateur et de NLP. Nos résultats montrent une amélioration significative par rapport aux méthodes de transfert learning classiques, notamment dans les scénarios à faibles données. Le graphique ci-dessous compare notre performance (en bleu) aux state-of-the-art existants.]
                </p>
                <img src="images/results_plot.png" alt="Graphique comparant les résultats">
                <figcaption>
                    <b>Figure 3 :</b> [Description du graphique de résultats.]
                </figcaption>
            </section>

            <!-- Section Citation (BibTeX) -->
            <section id="citation">
                <h2>Citation</h2>
                <p>Si vous trouvez notre travail utile pour vos recherches, veuillez considérer de citer notre papier :</p>
                <pre><code>@inproceedings{
    [VotreNom][Année][MotClé],
    author    = {[Votre Nom] and [Nom Co-auteur]},
    title     = {[Titre Complet de Votre Papier]},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {202X}
}</code></pre>
            </section>
        </main>

        <footer>
            <p>Site web conçu avec ❤️ pour ICML. Template par un assistant IA.</p>
        </footer>

    </div>

</body>
</html>
