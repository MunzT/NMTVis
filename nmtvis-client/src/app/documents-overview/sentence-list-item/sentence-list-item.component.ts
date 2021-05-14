import {Component, OnInit, AfterViewInit, Input, Output, OnChanges, SimpleChanges, SimpleChange, EventEmitter} from '@angular/core'
import {TextDisplayPipe} from '../../pipes/text-display.pipe';
import {DocumentService} from '../../services/document.service';
import {Router}                 from '@angular/router';
import {MetricInfo} from './metric-info'
import {DocumentsOverviewComponent} from '../documents-overview.component';

@Component({
    selector: 'app-sentence-list-item',
    templateUrl: './sentence-list-item.component.html',
    styleUrls: ['./sentence-list-item.component.css']
})
export class SentenceListItemComponent implements OnInit, OnChanges {

    @Input()
    sentence;
    @Input()
    selectedDocument;
    @Input()
    topics;
    @Input()
    selectedSentence;

    @Output()
    onSentenceSimilarityMetricChanged = new EventEmitter<any>();

    targetToBpe;
    bpeToSource;

    barChartVis;

    constructor(readonly router: Router, readonly documentService: DocumentService, private textPipe: TextDisplayPipe,
                private documentsOverviewComponent: DocumentsOverviewComponent) {
    }

    ngOnInit() {
        this.buildMappings();
    }

    ngAfterViewInit() {
        this.updateBarChart();
    }

    ngOnChanges(changes: SimpleChanges) {
        if (changes.topics && changes.topics.currentValue) {
            this.buildMappings();
        }
        if (changes.sentence && changes.sentence.currentValue) {
            this.sentence = changes.sentence.currentValue;
        }
        this.updateBarChart();
    }

    buildMappings() {
        let sentence = this.sentence;
        let bpe_target = sentence.translation.slice(0, -4).split(" ");
        let source = this.textPipe.transform(sentence.source, false).split(" ");
        let target = this.textPipe.transform(sentence.translation, true).split(" ");
        let bpe_source = sentence.source.split(" ");
        let attention = sentence.attention;

        var targetToBpe = {};
        var currentTargetIndex = 0;
        for (var j = 0; j < bpe_target.length; j++) {
            if (!(currentTargetIndex in targetToBpe)) {
                targetToBpe[currentTargetIndex] = [];
            }
            targetToBpe[currentTargetIndex].push(j);
            if (!bpe_target[j].endsWith('@@')) {
                currentTargetIndex++;
            }
        }
        this.targetToBpe = targetToBpe;

        var bpeToSource = {};
        var currentSourceIndex = 0;
        for (var j = 0; j < bpe_source.length; j++) {
            bpeToSource[j] = currentSourceIndex;

            if (!bpe_source[j].endsWith('@@')) {
                currentSourceIndex++;
            }
        }
        this.bpeToSource = bpeToSource;
    }

    openSentenceView(sentence) {
        this.router.navigate(['/document', this.selectedDocument.id, 'sentence', sentence.id]);
    }

    isHighlighted(word) {
        for (var topic of this.topics) {
            if (topic.active && word.trim().toLowerCase().indexOf(topic.name.toLowerCase()) >= 0) {
                return true;
            }
        }
        return false;
    }

    onSentenceClick(sentence, event) {
        event.stopPropagation();
        sentence.corrected = !sentence.corrected;

        this.documentService.setCorrected(this.selectedDocument.id, sentence.id, sentence.corrected)
            .subscribe(result => {
                if (sentence.flagged) {
                    sentence.flagged = false;
                    this.documentService.setFlagged(this.selectedDocument.id, sentence.id, sentence.flagged)
                        .subscribe(result => {
                        });
                }
            });
    }

    onSentenceFlag(sentence, event) {
        event.stopPropagation();
        sentence.flagged = !sentence.flagged;

        this.documentService.setFlagged(this.selectedDocument.id, sentence.id, sentence.flagged)
            .subscribe(result => {
            });
    }

    onFilter(sentence, event) {
        event.stopPropagation();
        var id = sentence.id;
        this.onSentenceSimilarityMetricChanged.emit(id);
    }

    isHighlightedTarget(sentence, word, target_index) {
        let bpe_target = sentence.translation.slice(0, -4).split(" ");
        let source = this.textPipe.transform(sentence.source, false).split(" ");
        let target = this.textPipe.transform(sentence.translation, true).split(" ");
        let bpe_source = sentence.source.split(" ");
        let attention = sentence.attention;

        var targetToBpe = this.targetToBpe
        var target_bpe_indices = targetToBpe[target_index];
        var source_bpe_indices = [];

        // Get all source bpe indices of affected words
        for (let target_bpe_index of target_bpe_indices) {
            for (let j = 0; j < attention[target_bpe_index].length; j++) {
                if (attention[target_bpe_index][j] > 0.3) {
                    if (source_bpe_indices.indexOf(j) < 0) {
                        source_bpe_indices.push(j);
                    }
                }
            }
        }

        var bpeToSource = this.bpeToSource;

        var source_indices = [];
        for (let source_bpe_index of source_bpe_indices) {
            source_indices.push(bpeToSource[source_bpe_index]);
        }

        for (let source_index of source_indices) {
            if (source[source_index] && this.isHighlighted(source[source_index])) {
                return true;
            }
        }
        return false;
    }

    updateBarChart() {
        if (!this.barChartVis) {
            this.barChartVis = new MetricInfo(this);
            this.barChartVis.build();
        } else {
            this.barChartVis.update();
        }
    }

}
