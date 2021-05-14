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

@Component({
    selector: 'app-sentence-view',
    templateUrl: './sentence-view.component.html',
    styleUrls: ['./sentence-view.component.css']
})

export class SentenceViewComponent implements OnInit, AfterContentInit {
    title = 'DNN Vis';
    ATTENTION_THRESHOLD = 0.2;

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
    beamSize = 3;
    beamLength = 1;
    beamCoverage = 1;
    sentenceId;
    documentId;
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
                    that.updateTranslation(sentence.inputSentence, sentence.translation);
                    that.updateBeamGraph(sentence.beam);
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
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            attentionOverrideMap: this.attentionOverrideMap,
            correctionMap: this.correctionMap,
            unk_map: this.unkMap
        }).subscribe(data => {
            this.updateBeamGraph(data["beam"]);
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
            beam_length: this.beamLength,
            beam_coverage: this.beamCoverage,
            unk_map: this.unkMap,
        }).subscribe(data => {
            this.updateBeamGraph(data["beam"]);
        });
    }

    onCurrentAttentionChange() {
        for (var i = 0; i < this.beamAttention.length; i++) {
            if (this.beamAttention[i] > this.ATTENTION_THRESHOLD) {
                d3.select("#source-word-text-" + i).style("font-weight", "bold");
                d3.select("#source-word-text-" + i).style("text-decoration", "underline");
            }
            let opacity = this.beamAttention[i] < 0.1 ? 0 : Math.sqrt(this.beamAttention[i]);
            d3.select("#source-word-box-" + i).style("opacity", opacity);
        }
    }

    highlightTargetNode(i) {
        d3.select("#target-word-text-" + i).style("font-weight", "bold");
        d3.select("#target-word-text-" + i).style("text-decoration", "underline");
        d3.select("#target-word-box-" + i).style("opacity", 1);
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

            if (attention[i][j] > this.ATTENTION_THRESHOLD) {
                svg.select("#source-word-text-" + j).style("font-weight", "bold");
                svg.select("#source-word-text-" + j).style("text-decoration", "underline");
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
        sourceWords.push("EOS");

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
                if (attention[i][maxJ] > this.ATTENTION_THRESHOLD
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
                svg.selectAll("path")
                    .classed("fade-out", true);
                svg.selectAll("[source-id='" + i + "']")
                    .classed("attention-selected", true);

                that.clearAttention();
                that.addEvent("source-hover", d);
                svg.select("#source-word-text-" + i).style("font-weight", "bold");
                svg.select("#source-word-text-" + i).style("text-decoration", "underline");
                svg.select("#source-word-box-" + i).style("opacity", 1);

                for (var j = 0; j < attention.length; j++) {
                    svg.select("#target-word-box-" + j).style("opacity", Math.sqrt(attention[j][i]));
                    if (attention[j][i] > that.ATTENTION_THRESHOLD) {
                        svg.select("#target-word-text-" + j).style("font-weight", "bold");
                        svg.select("#target-word-text-" + j).style("text-decoration", "underline");
                    }

                }
            })
            .on("mouseout", function (d, i) {
                svg.selectAll(".fade-out").classed("fade-out", false);
                svg.select("#source-word-text-" + i).style("font-weight", "normal");
                svg.select("#source-word-text-" + i).style("text-decoration", "none");
                that.addEvent("source-hover-out", d);
                svg.selectAll('.attention-selected').classed("attention-selected", false);
                that.updateAttentionViewWeights(attention, svg);
                svg.selectAll(".target-word-text").style("font-weight", "normal");
                svg.selectAll(".target-word-text").style("text-decoration", "none");
            });

        sourceEnter.append("g")
            .attr("transform", function (d, i) {
                var x = xSourceValues[i];
                var y = topY;
                return "translate(" + x + "," + y + ")";
            })
            .append("text")
            .attr("transform", function (d) {
                var xScale = 1;
                return "scale(" + xScale + ",1)"
            })
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
                svg.selectAll("path")
                    .classed("fade-out", true);
                svg.selectAll("[target-id='" + i + "']")
                    .classed("attention-selected", true);

                that.clearAttention();
                svg.select("#target-word-text-" + i).style("font-weight", "bold");
                svg.select("#target-word-text-" + i).style("text-decoration", "underline");
                that.addEvent("target-hover", d);

                svg.select("#target-word-box-" + i).style("opacity", 1);
                if (attention.length > i) {
                    for (var j = 0; j < attention[i].length; j++) {

                        svg.select("#source-word-box-" + j).style("opacity", Math.sqrt(attention[i][j]));

                        if (attention[i][j] > that.ATTENTION_THRESHOLD) {
                            svg.select("#source-word-text-" + j).style("font-weight", "bold");
                            svg.select("#source-word-text-" + j).style("text-decoration", "underline");
                        }

                    }
                }
            })
            .on("mouseout", function (d, i) {
                svg.selectAll(".fade-out").classed("fade-out", false);
                svg.select("#target-word-text-" + i).style("font-weight", "normal");
                svg.select("#target-word-text-" + i).style("text-decoration", "none");
                that.addEvent("target-hover-out", d);
                svg.selectAll('.attention-selected').classed("attention-selected", false);
                that.updateAttentionViewWeights(attention, svg);
                svg.selectAll(".source-word-text").style("font-weight", "normal");
                svg.selectAll(".source-word-text").style("text-decoration", "none");
            });

        targetEnter.append("g")
            .attr("transform", function (d, i) {
                var x = xTargetValues[i];
                var y = bottomY;
                return "translate(" + x + "," + y + ")";
            })
            .append("text")
            .attr("transform", function (d) {
                var xScale = 1;
                return "scale(" + xScale + ",1)"
            })
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

        tr.each(function (d, i) {
            if (!d) {
                return;
            }
        })

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
                return d < that.ATTENTION_THRESHOLD ? "hidden" : "visible";
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
        }

        if (m == 0) {
            for (var j = 0; j < this.beamAttention.length; j++) {
                 svg.select("#target-word-box-" + j).style("opacity", 0);
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
        this.documentService.retranslateSentence(this.documentId, this.sentenceId, this.beamSize).subscribe(res => {
            this.documentService.getSentence(this.documentId, this.sentenceId)
                            .subscribe((sentence: any) => {
                                this.sentence = sentence.inputSentence.split(" ");
                                this.inputSentence = this.decodeText(sentence.inputSentence);

                                this.translation = sentence.translation.split(" ");
                                this.attention = sentence.attention;
                                this.documentUnkMap = sentence.document_unk_map;
                                this.updateTranslation(sentence.inputSentence, sentence.translation);
                                this.updateBeamGraph(sentence.beam, true);
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
        this.documentService.getSentence(this.documentId, this.sentenceId)
            .subscribe((sentence: any) => {
                this.sentence = sentence.inputSentence.split(" ");
                this.inputSentence = this.decodeText(sentence.inputSentence);

                this.translation = sentence.translation.split(" ");
                this.attention = sentence.attention;
                this.documentUnkMap = sentence.document_unk_map;
                this.updateTranslation(sentence.inputSentence, sentence.translation);
                this.updateBeamGraph(sentence.beam, true);
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
