import {Component, OnInit, AfterContentInit, Inject} from '@angular/core';
import {MatDialog, MatDialogRef, MAT_DIALOG_DATA} from '@angular/material';
import {HttpClient} from '@angular/common/http';
import {ActivatedRoute, Router} from '@angular/router';
import * as d3 from 'd3';
import {BeamNode} from './beam-node';
import {BeamTree} from './beam-tree';
import {DocumentService} from '../services/document.service';
import {ExperimentService} from '../services/experiment.service';
import {MatSnackBar} from '@angular/material';
import {Constants} from '../constants';
import { MatFormFieldModule, MatSelectModule, MatInputModule } from '@angular/material';

@Component({
    selector: 'app-sentence-view',
    templateUrl: './sentence-view.component.html',
    styleUrls: ['./sentence-view.component.css']
})

export class SentenceViewComponent implements OnInit, AfterContentInit {
    attentionThreshold = 0.2;
    beamSize = 3;
    showMatrix = false;

    layerOptions = ["0", "1", "2", "3", "4", "5", "average"];
    attLayer = "4"
    showLayeroptions = false

    prev_beam_size = this.beamSize
    prev_attLayer = this.attLayer
    prev_attentionThreshold = this.attentionThreshold

    sentence = [];
    translation = [];
    editedTranslation;
    curr = [];
    heatmap = "";
    inputSentence = "";
    attention = [];
    loading = false;
    haveContent = false;
    sliderValues = [{word: "test", value: 0.2}, {word: "apple", value: 0.4}]
    translationIndex = 0;
    beamAttention = []
    partial = "";
    attentionOverrideMap = {};
    correctionMap = {};
    unkMap = {};
    documentUnkMap = {};
    beamLength = 1;
    beamCoverage = 1;
    sentenceId;
    documentId;
    showAttentionMatrix = true;
    beamTree;
    beam;
    objectKey = Object.keys;
    sideOpened = false;
    debug = false;
    events = [];

    interval;

    experimentMetrics: any = {
        timeSpent: 0,
        clicks: 0,
        hovers: 0,
        corrections: 0,
    };

    baseUrl = "http://localhost:5000";

    constructor(private http: HttpClient, private route: ActivatedRoute, private router: Router,
                private documentService: DocumentService, private experimentService: ExperimentService,
                public dialog: MatDialog, public snackBar: MatSnackBar) {
    }

    ngOnInit() {
        this.route.paramMap.subscribe(params => {
            this.documentId = params.get("document_id");
            this.sentenceId = params.get("sentence_id");

            this.unkMap = {};
            this.correctionMap = {};
            this.experimentMetrics = {
                timeSpent: 0,
            };
            this.events = [];

            var that = this;
            this.documentService.getSentence(this.documentId, this.sentenceId)
                .subscribe((sentence: any) => {
                    that.sentence = sentence.inputSentence.split(" ");
                    that.inputSentence = that.decodeText(sentence.inputSentence);

                    that.translation = sentence.translation.split(" ");
                    that.attention = sentence.attention;
                    that.documentUnkMap = sentence.document_unk_map;
                    that.showLayeroptions = (sentence.model === "trafo");
                    that.updateBeamGraph(sentence.beam);
                    that.beamSize = that.beamTree.countNumHypothesis();
                    that.prev_beam_size = that.beamSize;
                    that.prev_attLayer = that.attLayer
                    that.prev_attentionThreshold = that.attentionThreshold
                    that.updateAttentionMatrix(sentence.inputSentence, sentence.translation);
                    that.updateTranslation(sentence.inputSentence, sentence.translation);
                });
        });
    }

    ngAfterContentInit() {
        this.interval = setInterval(() => {
            this.experimentMetrics.timeSpent += 1;
        }, 500);
    }

    beamSizeChange() {
        this.http.post(this.baseUrl + '/beamUpdate', {
            sentence: this.encodeText(this.inputSentence),
            beam_size: this.beamSize,
            attLayer : Number(this.attLayer),
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            attentionOverrideMap: this.attentionOverrideMap,
            correctionMap: this.correctionMap,
            unk_map: this.unkMap
        }).subscribe(data => {
            this.updateBeamGraph(data["beam"], true);
            this.updateAttentionViewWeights(this.attention.slice(), d3);
        });
    }

    showMatrixChanged() {
        this.updateAttentionMatrix(this.sentence.join(" "), this.translation.join(" "));
    }

    attentionThresholdChange() {
        this.documentService.getSentence(this.documentId, this.sentenceId)
        .subscribe((sentence: any) => {
            this.sentence = sentence.inputSentence.split(" ");
            this.inputSentence = this.decodeText(sentence.inputSentence);

            this.translation = sentence.translation.split(" ");
            this.attention = sentence.attention;
            this.documentUnkMap = sentence.document_unk_map;
            this.updateTranslation(sentence.inputSentence, sentence.translation);
        });
    }

    layerChange() {
        this.http.post(this.baseUrl + '/layerUpdate', {
            sentence: this.encodeText(this.inputSentence),
            beam_size: this.beamSize,
            attLayer : Number(this.attLayer),
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            attentionOverrideMap: this.attentionOverrideMap,
            correctionMap: this.correctionMap,
            unk_map: this.unkMap
        }).subscribe(data => {
            this.documentService.getSentence(this.documentId, this.sentenceId)
                .subscribe((sentence: any) => {
                    this.sentence = sentence.inputSentence.split(" ");
                    this.inputSentence = this.decodeText(sentence.inputSentence);

                    this.translation = sentence.translation.split(" ");
                    this.attention = sentence.attention;
                    this.documentUnkMap = sentence.document_unk_map;
                    this.updateBeamGraph(data["beam"], true);
                    this.updateAttentionViewWeights(this.attention.slice(), d3);
                });
        });
    }

    attentionChange(event, i) {
        var changedValue = event.value;
        var restValue = (1.0 - changedValue) / this.beamAttention.length;

        for (var j = 0; j < this.beamAttention.length; j++) {
            if (j != i) {
                this.beamAttention[j] = restValue;
            }
        }
    }

    onCorrectionChange(word) {
        this.http.post(this.baseUrl + '/wordUpdate', {
            sentence: this.encodeText(this.inputSentence),
            attentionOverrideMap: this.attentionOverrideMap,
            correctionMap: this.correctionMap,
            beam_size: this.beamSize,
            attLayer : Number(this.attLayer),
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            unk_map: this.unkMap,
        }).subscribe(data => {
            this.updateBeamGraph(data["beam"]);
        });
    }

    onAttentionChange() {
        this.http.post(this.baseUrl + '/attentionUpdate', {
            sentence: this.encodeText(this.inputSentence),
            attentionOverrideMap: this.attentionOverrideMap,
            correctionMap: this.correctionMap,
            beam_size: this.beamSize,
            attLayer : this.attLayer,
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            unk_map: this.unkMap,
        }).subscribe(data => {
            this.updateBeamGraph(data["beam"]);
        });
    }

    attentionMouseOverSource(d, i, attention) {
        var svg = d3.select("#attention_vis");

        svg.selectAll("path")
            .classed("fade-out", true);
        svg.selectAll("[source-id='" + i + "']")
            .classed("attention-selected", true);

        this.clearAttention();
        this.addEvent("source-hover", d);
        svg.select("#source-word-text-" + i).style("font-weight", "bold");
        svg.select("#source-word-text-matrix-" + i).style("font-weight", "bold");
        svg.select("#source-word-text-" + i).style("text-decoration", "underline");
        svg.select("#source-word-text-matrix-" + i).style("text-decoration", "underline");
        svg.select("#source-word-box-" + i).style("opacity", 1);
        svg.select("#source-word-box-matrix-" + i).style("opacity", 1);

        for (var j = 0; j < attention.length; j++) {
            svg.select("#target-word-box-" + j).style("opacity", Math.sqrt(attention[j][i]));
            svg.select("#target-word-box-matrix-" + j).style("opacity", Math.sqrt(attention[j][i]));
            if (attention[j][i] > this.attentionThreshold) {
                svg.select("#target-word-text-" + j).style("font-weight", "bold");
                svg.select("#target-word-text-" + j).style("text-decoration", "underline");
                svg.select("#target-word-text-matrix-" + j).style("font-weight", "bold");
                svg.select("#target-word-text-matrix-" + j).style("text-decoration", "underline");
            }
        }
    }

    attentionMouseOverTarget(d, i, attention) {
        var svg = d3.select("#attention_vis");

        svg.selectAll("path")
            .classed("fade-out", true);
        svg.selectAll("[target-id='" + i + "']")
            .classed("attention-selected", true);

        this.clearAttention();
        svg.select("#target-word-text-" + i).style("font-weight", "bold");
        svg.select("#target-word-text-" + i).style("text-decoration", "underline");
        svg.select("#target-word-text-matrix-" + i).style("font-weight", "bold");
        svg.select("#target-word-text-matrix-" + i).style("text-decoration", "underline");
        this.addEvent("target-hover", d);

        svg.select("#target-word-box-" + i).style("opacity", 1);
        svg.select("#target-word-box-matrix-" + i).style("opacity", 1);
        if (attention.length > i) {
            for (var j = 0; j < attention[i].length; j++) {

                svg.select("#source-word-box-" + j).style("opacity", Math.sqrt(attention[i][j]));
                svg.select("#source-word-box-matrix-" + j).style("opacity", Math.sqrt(attention[i][j]));

                if (attention[i][j] > this.attentionThreshold) {
                    svg.select("#source-word-text-" + j).style("font-weight", "bold");
                    svg.select("#source-word-text-" + j).style("text-decoration", "underline");
                    svg.select("#source-word-text-matrix-" + j).style("font-weight", "bold");
                    svg.select("#source-word-text-matrix-" + j).style("text-decoration", "underline");
                }
            }
        }
    }

    attentionMouseOverCell(d, i, element, attention) {
        var i2 = d3.select(element.parentNode).attr("row");

        var svg = d3.select("#attention_vis");

        this.clearAttention();

        svg.select("#target-word-text-" + i2).style("font-weight", "bold");
        svg.select("#target-word-text-" + i2).style("text-decoration", "underline");
        svg.select("#target-word-text-matrix-" + i2).style("font-weight", "bold");
        svg.select("#target-word-text-matrix-" + i2).style("text-decoration", "underline");

        svg.select("#source-word-text-" + i).style("font-weight", "bold");
        svg.select("#source-word-text-" + i).style("text-decoration", "underline");
        svg.select("#source-word-text-matrix-" + i).style("font-weight", "bold");
        svg.select("#source-word-text-matrix-" + i).style("text-decoration", "underline");

        for (var j = 0; j < attention.length; j++) {
            svg.select("#target-word-box-" + j).style("opacity", Math.sqrt(attention[j][i]));
            svg.select("#target-word-box-matrix-" + j).style("opacity", Math.sqrt(attention[j][i]));
        }

        if (attention.length > i) {
            for (var j = 0; j < attention[i2].length; j++) {
                svg.select("#source-word-box-" + j).style("opacity", Math.sqrt(attention[i2][j]));
                svg.select("#source-word-box-matrix-" + j).style("opacity", Math.sqrt(attention[i2][j]));
            }
        }
    }

    attentionMouseOut(d, i, attention) {
        var svg = d3.select("#attention_vis");

        svg.selectAll(".fade-out").classed("fade-out", false);
        svg.select("#source-word-text-" + i).style("font-weight", "normal");
        svg.select("#source-word-text-matrix-" + i).style("font-weight", "normal");
        svg.select("#source-word-text-" + i).style("text-decoration", "none");
        svg.select("#source-word-text-matrix-" + i).style("text-decoration", "none");
        svg.select("#target-word-text-" + i).style("font-weight", "normal");
        svg.select("#target-word-text-" + i).style("text-decoration", "none");
        svg.select("#target-word-text-matrix-" + i).style("font-weight", "normal");
        svg.select("#target-word-text-matrix-" + i).style("text-decoration", "none");
        this.addEvent("source-hover-out", d);
        this.addEvent("target-hover-out", d);
        svg.selectAll('.attention-selected').classed("attention-selected", false);
        this.updateAttentionViewWeights(attention, svg);
        svg.selectAll(".target-word-text").style("font-weight", "normal");
        svg.selectAll(".target-word-text").style("text-decoration", "none");
        svg.selectAll(".source-word-text").style("font-weight", "normal");
        svg.selectAll(".source-word-text").style("text-decoration", "none");
    }

    updateAttentionMatrix(source: string, translation: string) {

        if (!this.showMatrix) {
            d3.selectAll("#attention-matrix-vis").remove();
            return;
        }

        var that = this;

        var sourceWords_bpe = source.split(" ");
        var targetWords_bpe = translation.split(" ");
        var sourceWords = sourceWords_bpe.slice(0);;
        var targetWords = targetWords_bpe.slice(0);;

        var index = 0;
        for (var s of sourceWords_bpe) {
            sourceWords[index] = this.decodeText(s);
            index++;
        }
        index = 0;
        for (var t of targetWords_bpe) {
            targetWords[index] = this.decodeText(t);
            index++;
        }

        var textWidths = [];
        for (var s of sourceWords) {
          textWidths.push(Math.ceil(this.calculateTextWidth(s)));
        }

        var maxSourceWidth = Math.max(...textWidths);

        textWidths = [];
        for (var t of targetWords) {
          textWidths.push(Math.ceil(this.calculateTextWidth(t)));
        }
        var maxTargetWidth = Math.max(...textWidths);

        var margin = {top: maxSourceWidth + 20, right: 20, bottom: 10, left: maxTargetWidth + 20},
            width = sourceWords.length * 20,
            height = targetWords.length * 20;

        d3.selectAll("#attention-matrix-vis").remove();
        var svg = d3.select("#attention-matrix").append("svg")
            .attr("width", width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .attr("id", "attention-matrix-vis")
            .append("g")
            .attr("transform", "translate("
                + margin.left + "," + margin.top + ")");

        var range: any = ["#f9f9f9", "#ffa500"];
        var colorScale: any = d3.scaleLinear().domain([0, 1]).range(range);

        var attention = this.attention.slice();

        var sourceEnter = svg.append('g').selectAll('text').data(sourceWords).enter();

        sourceEnter.append("rect")
            .attr("transform", (d, i) => {
                    return "translate(" + (i * 20 + 3) + ", 0)" + "rotate(90)";
                })
            .attr("x", function (d, i) {
                return - 10 - that.calculateTextWidth(d);
            })
            .attr("y", function (d, i) {
                return -10 - 3;
            })
            .attr("width", function (d) {
                return that.calculateTextWidth(d);
            })
            .attr("height", 15)
            .classed("source-word-box", true)
            .attr("id", function (d, i) {
                return "source-word-box-matrix-" + i;
            })
            .on("mouseover", function (d, i) {
                that.attentionMouseOverSource(d, i, attention);
            })
            .on("mouseout", function (d, i) {
                that.attentionMouseOut(d, i, attention);
            });

        var targetEnter = svg.append('g').selectAll('text').data(targetWords).enter();

        targetEnter.append("rect")
            .attr("x", function (d, i) {
                return -that.calculateTextWidth(d) - 9;
            })
            .attr("y", function (d, i) {
                return i * 20 + 1;
            })
            .attr("width", function (d) {
                return that.calculateTextWidth(d);
            })
            .attr("height", 15)
            .classed("target-word-box", true)
            .attr("id", function (d, i) {
                return "target-word-box-matrix-" + i;
            })
            .on("mouseover", function (d, i) {
                that.attentionMouseOverTarget(d, i, attention);
            })
            .on("mouseout", function (d, i) {
                that.attentionMouseOut(d, i, attention);
            });

        sourceEnter.append("g")
            .attr("transform", (d, i) => {
                return "translate(" + (i * 20 + 6) + ", -10)" + "rotate(90)";
            })
            .append("text")
            .classed("source-word-text", true)
            .attr("id", function (d, i) {
                return "source-word-text-matrix-" + i;
            })
            .style("font-size", "12px")
            .style("text-anchor", "end")
            .text((d, i) => sourceWords[i]);

        targetEnter.append("g")
            .attr("transform", (d, i) => {
                return "translate(-10," + (i * 20 + 12) + ")";
            })
            .append("text")
            .classed("target-word-text", true)
            .attr("id", function (d, i) {
                return "target-word-text-matrix-" + i;
            })
            .style("font-size", "12px")
            .style("text-anchor", "end")
            .text((d, i) => (targetWords[i] === Constants.EOS ? "EOS" : targetWords[i]));

        var rowsEnter = svg.selectAll(".row")
            .data(attention)
            .enter();

        var rows = rowsEnter.append("g")
            .attr("class", "row")
            .attr("transform", (d, i) => {
                return "translate(0," + i * 20 + ")";
            })
            .attr("row", (d, i) => {
                return i;
            });
        var squares = rows.selectAll(".attention-cell")
            .data(d => d)
            .enter().append("rect")
            .attr("x", (d, i) => i * 20)
            .attr("width", 20 - 4)
            .attr("height", 20 - 4)
            .style("fill", d => colorScale(d))
            .on("mouseover", function (d, i) {
                that.attentionMouseOverCell(d, i, this, attention);
            })
            .on("mouseout", function (d, i) {
                that.attentionMouseOut(d, i, attention);
            });

        var sourceLines = svg.selectAll(".source-lines")
            .data(sourceWords_bpe)
            .enter()
            .append("line")
            .attr("class", "source-lines")
            .style("stroke", "black")
            .style("stroke-width", 3)
            .attr("y1", -2)
            .attr("x1", (d, i) => (i * 20))
            .attr("y2", -2)
            .attr("x2", (d, i) => (i * 20 + 20 + (d.endsWith("@@") ? + 1 : -4)));

        var sourceLines = svg.selectAll(".source-lines-2")
            .data(sourceWords_bpe)
            .enter()
            .append("line")
            .attr("class", "source-lines-2")
            .style("stroke", "#777")
            .style("stroke-width", 1)
            .attr("y1", height - 4)
            .attr("x1", (d, i) => ((i + 1) * 20 - 2))
            .attr("y2", -20)
            .attr("x2", (d, i) => ((i + 1) * 20 - 2))
            .style("opacity", (d, i) => d.endsWith("@@") || (i == sourceWords_bpe.length - 1) ? 0 : 1);

        var targetLines = svg.selectAll(".target-lines")
            .data(targetWords_bpe)
            .enter()
            .append("line")
            .attr("class", "target-lines")
            .style("stroke", "black")
            .style("stroke-width", 3)
            .attr("y1", (d, i) => (i * 20))
            .attr("x1", -2)
            .attr("y2", (d, i) => (i * 20 + 20 + (d.endsWith("@@") ? + 1 : -4)))
            .attr("x2", -2);

        var targetLines = svg.selectAll(".target-lines-2")
            .data(targetWords_bpe)
            .enter()
            .append("line")
            .attr("class", "target-lines-2")
            .style("stroke", "#777")
            .style("stroke-width", 1)
            .attr("y1", (d, i) => ((i + 1) * 20 - 2))
            .attr("x1", width - 4)
            .attr("y2", (d, i) => ((i + 1) * 20 - 2))
            .attr("x2", -20)
            .style("opacity", (d, i) => d.endsWith("@@") || (i == targetWords_bpe.length - 1) ? 0 : 1);

        that.updateAttentionViewWeights(attention, svg);
    }

    onCurrentAttentionChange() {
        for (var i = 0; i < this.beamAttention.length; i++) {
            if (this.beamAttention[i] > this.attentionThreshold) {
                d3.select("#source-word-text-" + i).style("font-weight", "bold");
                d3.select("#source-word-text-" + i).style("text-decoration", "underline");
                d3.select("#source-word-text-matrix-" + i).style("font-weight", "bold");
                d3.select("#source-word-text-matrix-" + i).style("text-decoration", "underline");
            }
            let opacity = this.beamAttention[i] < 0.1 ? 0 : Math.sqrt(this.beamAttention[i]);
            d3.select("#source-word-box-" + i).style("opacity", opacity);
            d3.select("#source-word-box-matrix-" + i).style("opacity", opacity);
        }
    }

    highlightTargetNode(i) {
        d3.select("#target-word-text-" + i).style("font-weight", "bold");
        d3.select("#target-word-text-" + i).style("text-decoration", "underline");
        d3.select("#target-word-text-matrix-" + i).style("font-weight", "bold");
        d3.select("#target-word-text-matrix-" + i).style("text-decoration", "underline");
        d3.select("#target-word-box-" + i).style("opacity", 1);
        d3.select("#target-word-box-matrix-" + i).style("opacity", 1);
    }

    clearAttentionSelection() {
        var attention = this.attention.slice();
        var svg = d3.select("#translation-vis");
        this.updateAttentionViewWeights(attention, svg);

        d3.selectAll(".source-word-text").style("font-weight", "normal");
        d3.selectAll(".source-word-text").style("text-decoration", "none");
        d3.selectAll(".target-word-text").style("font-weight", "normal");
        d3.selectAll(".target-word-text").style("text-decoration", "none");
    }

    clearAttention() {
        d3.selectAll(".source-word-box").style("opacity", 0);
        d3.selectAll(".target-word-box").style("opacity", 0);

        d3.selectAll(".source-word-text").style("font-weight", "normal");
        d3.selectAll(".source-word-text").style("text-decoration", "none");
        d3.selectAll(".target-word-text").style("font-weight", "normal");
        d3.selectAll(".target-word-text").style("text-decoration", "none");
    }

    calculateTextWidth(text) {
        text = text.replace("@@", "")

        var svg = !d3.select("#translation-vis").empty() ?
            d3.select("#translation-vis") : d3.select("body").append("svg").attr("id", "sample");
        var textSel: any = svg.append("text").text(text).style("font-size", "12px");
        var width = textSel.node().getComputedTextLength();
        textSel.remove();
        d3.select("#sample").remove();
        return width + 2;
    }

    mouseoverTargetWord(i, attention) {
        var svg = d3.select("#translation-vis");
        svg.selectAll("path").classed("fade-out", true);
        svg.selectAll("[target-id='" + i + "']")
            .classed("attention-selected", true);

        if (!attention[i]) {
            return;
        }

        for (var j = 0; j < attention[i].length; j++) {

            svg.select("#source-word-box-" + j).style("opacity", Math.sqrt(attention[i][j]));
            svg.select("#source-word-box-matrix-" + j).style("opacity", Math.sqrt(attention[i][j]));

            if (attention[i][j] > this.attentionThreshold) {
                svg.select("#source-word-text-" + j).style("font-weight", "bold");
                svg.select("#source-word-text-" + j).style("text-decoration", "underline");
                svg.select("#source-word-text-matrix-" + j).style("font-weight", "bold");
                svg.select("#source-word-text-matrix-" + j).style("text-decoration", "underline");
            }
        }
    }

    isValidTranslation() {
        return this.translation.length !== 0 && this.translation[this.translation.length - 1] === Constants.EOS;
    }

    lastMaxIndex(a) {
        var maxI = 0;
        for (var i = 1; i < a.length; i++) {
            if (a[i] >= a[maxI]) {
                maxI = i;
            }
        }
        return maxI;
    }

    updateTranslation(source: string, translation: string) {

        var that = this;
        var textWidth = 70;
        var leftMargin = 120;

        var maxTextLength = 10;
        var barPadding = 1;
        var targetBarPadding = 1;

        var attention = this.attention.slice();

        var sourceWords = source.split(" ");
        //sourceWords.push("EOS");

        var targetWords = translation.split(" ");

        var xTargetValues = {0: 0};
        var wholeWordCount = 0;
        var furthestRelevantSourceIndex = 0;

        for (var i = 1; i < targetWords.length; i++) {
            xTargetValues[i] = xTargetValues[i - 1] + 0.5 * this.calculateTextWidth(targetWords[i])
                + 0.5 * this.calculateTextWidth(targetWords[i - 1]) + barPadding;
            if (!targetWords[i - 1].endsWith("@@")) {
                xTargetValues[i] += 3;
                wholeWordCount++;
            }
        }
        for (var i = 1; i < attention.length; i++) {
           for (var j = 0; j < attention[i].length; j++) {
                var maxJ = this.lastMaxIndex(attention[i]);
                if (attention[i][maxJ] > this.attentionThreshold
                    && (maxJ !== attention[i].length - 1 || i === this.inputSentence.split(" ").length - 1)) {
                    furthestRelevantSourceIndex = Math.max(furthestRelevantSourceIndex, maxJ);
                }
            }
        }

        var xSourceValues = {0: 0};
        for (var i = 1; i < sourceWords.length; i++) {
            xSourceValues[i] = xSourceValues[i - 1] + 0.5 * this.calculateTextWidth(sourceWords[i])
                + 0.5 * this.calculateTextWidth(sourceWords[i - 1]) + barPadding;
            if (!sourceWords[i - 1].endsWith("@@")) {
                xSourceValues[i] += 3;
            }
        }

        var w1 = xSourceValues[sourceWords.length - 1] + 0.5 * this.calculateTextWidth(sourceWords.slice(-1)[0]);
        var w2 = xTargetValues[targetWords.length - 1] + 0.5 * this.calculateTextWidth(targetWords.slice(-1)[0]);
        var relevantW1 = xSourceValues[furthestRelevantSourceIndex] + 0.5 * this.calculateTextWidth(sourceWords[furthestRelevantSourceIndex]);

        if (wholeWordCount > 1 && targetWords.length > 2 && relevantW1 - (w2 + leftMargin) > 0) {
            let delta = relevantW1 - w2 - leftMargin;
            if (delta > 0) {
                targetBarPadding = delta / (wholeWordCount - 1)

                var wholeWordIndex = 0;
                for (var i = 0; i < targetWords.length; i++) {
                    xTargetValues[i] += wholeWordIndex * targetBarPadding;
                    if (!targetWords[i].endsWith("@@")) {
                        wholeWordIndex++;
                    }
                }
            }
        }

        var w = Math.max(w1, w2) + leftMargin;

        var margin = {top: 20, right: 20, bottom: 20, left: leftMargin},
            test_width = w - margin.left - margin.right,
            height = 100 - margin.top - margin.bottom;

        var attentionScale: any = d3.scalePow().domain([0, 1]).range([0, 5]);
        attentionScale = function (x) {
            return 6 * Math.sqrt(x);
        }

        // append the svg object to the body of the page
        // appends a 'group' element to 'svg'
        // moves the 'group' element to the top left margin
        d3.selectAll("#translation-vis").remove();
        var svg = d3.select("#translation-box").append("svg")
            .attr("width", test_width + margin.right + margin.left)
            .attr("height", height + margin.top + margin.bottom)
            .attr("id", "translation-vis")
            .append("g")
            .attr("transform", "translate("
                + margin.left + "," + margin.top + ")");
        var topY = 10;
        var bottomY = 70;

        var targetWords = translation.split(" ");

        svg.append('text').attr("y", topY).attr("x", -textWidth - 50).style("font-weight", "bold")
            .style("text-anchor", "left")
            .text("Source");
        svg.append('text').attr("y", bottomY).attr("x", -textWidth - 50).style("font-weight", "bold")
            .style("text-anchor", "left")
            .text("Translation");

        var sourceEnter = svg.append('g').selectAll('text').data(sourceWords).enter();

        sourceEnter.append("rect")
            .attr("x", function (d, i) {
                return xSourceValues[i] - 0.5 * that.calculateTextWidth(d);
            })
            .attr("y", function (d, i) {
                return topY - 15;
            })
            .attr("width", function (d) {
                return that.calculateTextWidth(d);
            })
            .attr("height", 20)
            .classed("source-word-box", true)
            .attr("id", function (d, i) {
                return "source-word-box-" + i;
            })
            .on("mouseover", function (d, i) {
                that.attentionMouseOverSource(d, i, attention)
            })
            .on("mouseout", function (d, i) {
                that.attentionMouseOut(d, i, attention);
            });

        sourceEnter.append("g")
            .attr("transform", function (d, i) {
                var x = xSourceValues[i];
                var y = topY;
                return "translate(" + x + "," + y + ")";
            })
            .append("text")
            .classed("source-word-text", true)
            .attr("id", function (d, i) {
                return "source-word-text-" + i;
            })
            .text(function (d) {
                return that.decodeText(d);
            })
            .style("text-anchor", "middle");

        var targetEnter = svg.append('g').selectAll('text').data(targetWords).enter();

        targetEnter.append("rect")
            .attr("x", function (d, i) {
                return xTargetValues[i] - 0.5 * that.calculateTextWidth(d);
            })
            .attr("y", function (d, i) {
                return bottomY - 15;
            })
            .attr("width", function (d) {
                return that.calculateTextWidth(d);
            })
            .attr("height", 20)
            .classed("target-word-box", true)
            .attr("id", function (d, i) {
                return "target-word-box-" + i;
            })
            .on("mouseover", function (d, i) {
                that.attentionMouseOverTarget(d, i, attention);
            })
            .on("mouseout", function (d, i) {
                that.attentionMouseOut(d, i, attention);
            });

        targetEnter.append("g")
            .attr("transform", function (d, i) {
                var x = xTargetValues[i];
                var y = bottomY;
                return "translate(" + x + "," + y + ")";
            })
            .append("text")
            .classed("target-word-text", true)
            .attr("id", function (d, i) {
                return "target-word-text-" + i;
            })
            .text(function (d) {
                if (d === Constants.EOS) {
                    return "EOS";
                }
                return that.decodeText(d);
            })
            .style("text-anchor", "middle");

        var tr = svg.append('g').selectAll('g').data(attention).enter().append("g");

        var j = -1;
        tr.selectAll('path').data(function (d) {
            return d;
        })
            .enter()
            .append("path")
            .classed("attention-line", true)
            .attr("d", function (d, i) {
                if (i == 0) {
                    j++;
                }

                d3.select(this).attr('source-id', i + "");
                d3.select(this).attr('target-id', j + "");

                var pos: any = [{x: xSourceValues[i], y: topY + 5}, {x: xSourceValues[i], y: topY + 15},
                    {
                        x: (xTargetValues[j] + xSourceValues[i]) / 2,
                        y: (topY + bottomY - 15) / 2
                    }, {
                        x: xTargetValues[j],
                        y: bottomY - 25,
                    }, {
                        x: xTargetValues[j],
                        y: bottomY - 15
                    }];
                var line = d3.line().curve(d3.curveBundle.beta(1))
                    .x(function (d: any) {
                        return d.x;
                    })
                    .y(function (d: any) {
                        return d.y;
                    });
                return line(pos);
            })
            .style("stroke-width", function (d) {
                return attentionScale(d) + "px";
            })
            .style("visibility", function (d, i) {
                return d < that.attentionThreshold ? "hidden" : "visible";
            });

        that.updateAttentionViewWeights(attention, svg);
    }

    updateAttentionViewWeights(attention, svg) {
        var m = attention.length;
        var n = Math.max(attention[0] ? attention[0].length : 0, attention[1] ? attention[1].length : 0); // attention[0] sometimes different...
        var max = 0;
        svg.selectAll(".source-word-box").style("opacity", 0);
        for (var j = 0; j < n; j++) {
            var sum = 0
            for (var k = 0; k < m; k++) {
                sum += attention[k] && attention[k][j] ? Math.abs(attention[k][j]) : 0;
            }
            max = Math.max(max, Math.abs(sum));
            svg.select("#source-word-box-" + j).style("opacity", sum);
            svg.select("#source-word-box-matrix-" + j).style("opacity", sum);
        }

        if (m == 0) {
            for (var j = 0; j < this.beamAttention.length; j++) {
                 svg.select("#target-word-box-" + j).style("opacity", 0);
                 svg.select("#target-word-box-matrix-" + j).style("opacity", 0);
            }
        }
        max = 0;
        for (var k = 0; k < m; k++) {
            var sum = 0
            for (var j = 0; j < n; j++) {
                sum += attention[k] && attention[k][j] ? Math.abs(attention[k][j]) : 0;
            }
            max = Math.max(max, Math.abs(sum));
            svg.select("#target-word-box-" + k).style("opacity", sum);
            svg.select("#target-word-box-matrix-" + k).style("opacity", sum);
        }
    }

    encodeText(text) {
        return text.replace(/'/g, "&apos;").replace(/"/g, '&quot;')
            .replace(/\u200b\u200b/g, "@@ ")
            .replace(/\u200b/g, "@@");
    }

    decodeText(text) {
        return text.replace(/&apos;/g, "'").replace(/&quot;/g, '"')
            .replace(/@@ /g, "\u200b\u200b")
            .replace(/@@/g, "\u200b");
    }

    updateBeamGraph(treeData, highlightLastNode = false) {
        this.beam = treeData;
        if (!this.beamTree) {
            this.beamTree = new BeamTree(treeData, this, this.documentService);
            this.beamTree.build();
        } else {
            this.beamTree.updateData(treeData, highlightLastNode);
        }
    }

    onSkip() {
        this.router.navigate(['/documents', this.documentId, "sentence", this.sentenceId]);
    }

    onRetranslate() {
        this.unkMap = {};
        this.correctionMap = {};
        this.experimentMetrics = {
            timeSpent: 0,
        };
        this.events = [];


        this.documentService.retranslateSentence(this.documentId, this.sentenceId, this.beamSize, Number(this.attLayer)).subscribe(res => {
            this.documentService.getSentence(this.documentId, this.sentenceId)
                            .subscribe((sentence: any) => {
                                this.sentence = sentence.inputSentence.split(" ");
                                this.inputSentence = this.decodeText(sentence.inputSentence);

                                this.translation = sentence.translation.split(" ");
                                this.attention = sentence.attention;
                                this.documentUnkMap = sentence.document_unk_map;
                                this.prev_beam_size = this.beamSize;
                                this.prev_attLayer = this.attLayer
                                this.prev_attentionThreshold = this.attentionThreshold

                                this.updateBeamGraph(sentence.beam, true);
                                this.updateAttentionMatrix(sentence.inputSentence, sentence.translation);
                                this.updateTranslation(sentence.inputSentence, sentence.translation);
                    });
                });
    }

    onReset() {
        this.unkMap = {};
        this.correctionMap = {};
        this.experimentMetrics = {
            timeSpent: 0,
        };
        this.events = [];
        this.beamSize = this.prev_beam_size
        this.attLayer = this.prev_attLayer
        this.attentionThreshold = this.prev_attentionThreshold

        this.documentService.getSentence(this.documentId, this.sentenceId)
            .subscribe((sentence: any) => {
                this.sentence = sentence.inputSentence.split(" ");
                this.inputSentence = this.decodeText(sentence.inputSentence);

                this.translation = sentence.translation.split(" ");
                this.attention = sentence.attention;
                this.documentUnkMap = sentence.document_unk_map;

                this.updateBeamGraph(sentence.beam, true);
                this.updateAttentionMatrix(sentence.inputSentence, sentence.translation);
                this.updateTranslation(sentence.inputSentence, sentence.translation);

            });
    }

    onAcceptTranslation() {
        this.beamTree.mouseover(this.beamTree.lastNodeOfGoldenHypothesis(this.beamTree.root),
            this.beamTree.getNodeSelection(this.beamTree.lastNodeOfGoldenHypothesis(this.beamTree.root)));
        this.beamTree.center(this.beamTree.lastNodeOfGoldenHypothesis(this.beamTree.root), "");
        this.http.post(this.baseUrl + '/api/correctTranslation', {
            translation: this.translation.join(" "),
            beam: this.beam,
            attention: this.attention,
            document_id: this.documentId,
            sentence_id: this.sentenceId,
            document_unk_map: this.documentUnkMap
        }).subscribe(data => {
            let snackBarRef = this.snackBar.open('Translation accepted!', '', {duration: 700});
            this.router.navigate(['/documents', this.documentId, "sentence", this.sentenceId]);
        });
    }

    addEvent(type, val = "") {
        this.events.push({"type": type, "time": this.experimentMetrics.timeSpent, "val": val})
    }

    onTranslationEdit($event) {
        this.editedTranslation = this.encodeText($event);
        if (this.editedTranslation[this.editedTranslation.length - 1] === " ") {

            var prefix = (Constants.SOS + " " + this.editedTranslation.trim()).split(" ");
            this.correctionMap = {};
            this.correctionMap[prefix.slice(0, prefix.length - 1).join(" ")] = prefix[prefix.length - 1];
            this.onCorrectionChange("");
        }
    }

    onClick() {
        this.loading = true;
        this.correctionMap = {};
        this.attentionOverrideMap = {};
        this.http.post(this.baseUrl, {
            sentence: this.encodeText(this.inputSentence),
            beam_size: this.beamSize,
            attLayer : Number(this.attLayer),
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
        }).subscribe(data => {
            this.sentence = data["sentence"].split(" ");
            this.translation = this.decodeText(data["translation"]).split(" ");
            this.attention = data["attention"];
            this.beamAttention = [1, 0, 0, 0]
            this.loading = false;
            this.haveContent = true;
            this.updateBeamGraph(data["beam"]);
            this.updateAttentionMatrix(data["sentence"], data["translation"]);
            this.updateTranslation(data["sentence"], data["translation"]);
        });
    }

    showInfo() {
        this.dialog.open(InfoDialog, {
            width: "600px"
        });
    }

    mouseEnter(event) {
        this.curr = this.attention[event];
    }

    getColor(i) {
        let colors = ["#ffcdd2", "#ef9a9a", "#e57373", "#EF5350", "#F44336", "#E53935", "#d32f2f"]
        let index = Math.round(this.curr[i] * (colors.length - 1));

        return colors[index];
    }
}

@Component({
    selector: 'info-dialog',
    templateUrl: 'info-dialog.html',
})
export class InfoDialog {

    shownValues = [];
    beamAttention = [];

    constructor(public dialogRef: MatDialogRef<InfoDialog>,
                @Inject(MAT_DIALOG_DATA) public data: any) {
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}

@Component({
    selector: 'beam-node-dialog',
    templateUrl: 'beam-node-dialog.html',
})
export class BeamNodeDialog {

    shownValues = [];
    beamAttention = [];
    events = [];
    sentenceView;

    constructor(public dialogRef: MatDialogRef<BeamNodeDialog>,
                @Inject(MAT_DIALOG_DATA) public data: any) {
        this.beamAttention = data.attention;
        this.events = data.events;
        this.shownValues = this.beamAttention.slice();
        this.sentenceView = data.sentenceView;
    }

    onKeyDown(event) {
        this.sentenceView.addEvent("keydown", event.key);
    }

    onAttentionChange() {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}
