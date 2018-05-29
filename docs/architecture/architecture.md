# Architektur - FastText on distributed TensorFlow

## FastText

## Distribution

Für die Verteilung der Berechnung der Word-Vektoren auf mehrere Rechner, wird der in TensorFlow vorgesehene Mechanismus genutzt. Dabei werden die zur Verfügung stehenden Server aufgeteilt in Parameter-Server und Worker.  
- Die Parameter-Server speichern die Variablen des TensorFlow-Graphen. Braucht ein Worker für die aktuelle Berechnung den Wert einer Variable so wird dieser vom betreffenden Parameter-Server an den Worker übermittelt. Da bei einer sehr großen Menge von Variablen Speicher oder Netzwer-IO eines einzelnen Parameter-Servers eventuell nicht mehr ausreichen ist es auch möglich die Variablen unter mehreren Parameter Serven aufzuteilen. Für große Trainingsdatensätze und hohe Embedding-Dimensionen dürfte das auch in unserem Fall nötig sein. 
- Die Worker führen die eigentlichen Berechnungen durch, in unserem Fall also die Berechnung von Word-Vektoren aus Trainingsdaten. Jeder Worker operiert dabei auf einem Teil der Trainingsdaten. Bestimmte Daten, wie das Mapping Wort->int oder die Worthäufigkeiten in den Trainingsdaten müssen allerdings auf allen Workern vorliegen. Dies kann entweder über die seperate Berechnung auf jedem Worker oder die einmalige Berechnung und anschließende Verteilung über die Parameter-Server erfolgen. Welche der beiden Varianten performanter ist wird sich zeigen.  

Die benötigten Daten-Übertragungen zwischen Workern und Parameter-Servern werden von TensorFlow automatisch in den Berechnungs-Graphen eingefügt. Von unserer Seite ist dafür keine zusätzlicher Arbeit nötig. 

Der Grobe Ablauf des Trainings sieht folgendermaßen aus:  

0. Variablen werden initialisiert. Die beteiligten Server bauen Verbindungen zueinander auf und warten bis alle Server bereit sind.
1. Jeder Worker generiert aus seinem Teil der Trainingsdaten seinen nächsten Trainings-Batch.
2. Jeder Worker holt sich vom jeweils zuständigen Parameter-Server die aktuellen Vektoren für die n-gramme der Ziel-Worte, sowie die Vektoren der passenden Kontext-Worte und die Vektoren  von einer Menge zufälliger nicht-passender Kontext-Worte
3. Jeder Worker führt das NCE-Sampling für seine aktuelle Batch durch und berechnet die vorzunehmenden Änderungen an den Vektoren.
4. Jeder Worker übermittelt die berechneten Änderungen an den Vektoren asynchron an die Parameter-Server. (Die Asynchronität der Updates könnte zu suboptimalen Verhalten führen, laut der TensorFlow-Dokumentation kommt es aber zu keinenen praktisch-relevanten Problemen)
5. Wenn das Training nicht aufgrund von der vorgegebenen Schrittzahl oder Fehlerquote beendet wurde, gehe zu Schritt 1.




Hier ein kleines Diagramm um den (zugegeben recht simplen) Aufbau zu veranschaulichen:

![](./distribution-diagram.png)

## Deployment

Der aktuelle Plan sieht vor führ das Deployment der verteilten Umgebung auf Docker zurückzugreifen.  
Während der Entwicklungsphase wird docker-compose genutzt um auf einem einzelnen Rechner mehrere isolierte Instanzen der Software zu starten, die mittels eines virtuellen Netzwerkes miteinander kommunizieren.  
Für das Deployment auf dem Galaxy-Cluster soll docker-swarm benutzt werden, wodurch die verschiedenen Instanzen der Software auf mehrere physische Rechner verteilt und dennoch zentral verwaltet werden können. Laut der Webseite des Uni-Rechenzentrums ist Docker auf den Nodes des Galaxy-Clusters bereits installiert.