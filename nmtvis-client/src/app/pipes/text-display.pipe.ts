import {Pipe, PipeTransform} from '@angular/core';
import {Constants} from '../constants';

@Pipe({
    name: 'textDisplay'
})
export class TextDisplayPipe implements PipeTransform {

    transform(text: string, removeEOS: boolean): any {
        text = text.replace(/&apos;/g, "'");
        text = text.replace(/&quot;/g, '"');
        text = text.replace(/& quot ;/g, '"');
        text = text.replace(/@@ /g, '');
        text = text.replace(/@@/g, '');
        text = text.replace(/\u200b\u200b/, "")
        text = text.replace(/\u200b\u200b/, "")
        text = text.replace(/\u200b /, "")
        let rest = removeEOS ? text.slice(1, -(Constants.EOS.length + 1)) : text.slice(1);

        return text.slice(0, 1).toUpperCase() + rest;
    }

}
